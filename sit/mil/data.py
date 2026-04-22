import numpy as np
from dataclasses import dataclass, field


# the output of MILData function is a MILData object
# it contains the following fields:
# X: the instance embeddings
# y: the bag labels
# group_sizes: the number of instances in each bag
# instance_y: the instance labels

@dataclass
class MILData:
    X: np.ndarray
    y: np.ndarray | None
    group_sizes: np.ndarray
    group_ids: np.ndarray | None = field(default=None)
    shifts: np.ndarray | None = field(default=None)
    instance_y: np.ndarray | None = field(default=None)

    # the meaning of shifts: shifts[i] is the index of the first instance in the i-th bag
    # so the i-th bag is X[shifts[i]:shifts[i + 1]] 
    # if the shifts are not provided, they are calculated from the group_sizes
    def __post_init__(self):
        if self.group_ids is None:
            self.group_ids = np.array([
                g_id
                for g_id, gs in enumerate(self.group_sizes)
                for _ in range(gs)
            ])
        if self.shifts is None:
            self.shifts = np.insert(np.cumsum(self.group_sizes), 0, 0)
        if self.instance_y is not None:
            assert len(self.instance_y) == len(self.X), 'instance_y should have one label per instance'

    @classmethod
    def from_bags(cls, bags, y=None, instance_y=None) -> 'MILData':
        if instance_y is not None and not isinstance(instance_y, np.ndarray):
            instance_y = np.concatenate([
                np.asarray(b)
                for b in instance_y
            ], axis=0)
        return cls(
            X=np.concatenate([
                np.asarray(b)
                for b in bags
            ], axis=0),
            y=y,
            group_sizes=np.array([len(b) for b in bags]),
            instance_y=instance_y,
        )

    def union(self, other: 'MILData') -> 'MILData':
        if self.instance_y is None and other.instance_y is None:
            instance_y = None
        else:
            assert self.instance_y is not None and other.instance_y is not None, \
                'Cannot union MILData with instance_y present in only one side'
            instance_y = np.concatenate([self.instance_y, other.instance_y], axis=0)
        return MILData(
            X=np.concatenate([self.X, other.X], axis=0),
            y=np.concatenate([self.y, other.y], axis=0),
            group_sizes=np.concatenate([self.group_sizes, other.group_sizes], axis=0),
            instance_y=instance_y,
        )

    def __getitem__(self, subset_ids) -> 'MILData':
        assert not str(subset_ids.dtype) == 'bool', 'Masks are not supported'
        group_sizes = self.group_sizes[subset_ids]
        assert self.shifts is not None
        return MILData(
            X=np.concatenate(
                [
                    self.X[self.shifts[i]:self.shifts[i + 1]]
                    for i in subset_ids
                ],
                axis=0
            ),
            y=(self.y[subset_ids] if self.y is not None else self.y),
            group_sizes=group_sizes,
            instance_y=(
                np.concatenate(
                    [
                        self.instance_y[self.shifts[i]:self.shifts[i + 1]]
                        for i in subset_ids
                    ],
                    axis=0
                )
                if self.instance_y is not None else None
            ),
        )

    def __len__(self) -> int:
        return len(self.group_sizes)
