import torch
from ..mil.data import MILData
from gradient_growing_trees.tree_nn import TreeNN
from gradient_growing_trees.tree import BatchArbitraryLoss
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone, defaultdict
from sklearn.metrics import r2_score
from abc import ABCMeta, abstractmethod


class AttentionAggregationNN(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, out_features: int = 1):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # query is a learnable parameter that is used to compute the attention logits
        self.query = torch.nn.Parameter(torch.ones((1, 1, embed_dim), dtype=torch.float32, requires_grad=True))
        self.linear = torch.nn.Linear(embed_dim, out_features)
        self.group_ids = None
        self.last_instance_embeddings = None
        self.last_attention_logits = None
        self.last_attention_weights = None

    def _recompute_group_cache(self, group_ids):
        group_ids = group_ids.reshape(-1)

        # Compare values rather than object identity.
        if (
            self.group_ids is not None
            and self.group_ids.device == group_ids.device
            and torch.equal(self.group_ids, group_ids)
        ):
            return

        self.group_ids = group_ids.detach().clone()

        unique_group_ids, dense_group_ids, group_sizes = torch.unique(
            group_ids,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

        if unique_group_ids.numel() == 0:
            raise ValueError("At least one instance is required")

        self.unique_group_ids = unique_group_ids
        self.n_groups = unique_group_ids.numel()
        self.max_group_size = int(group_sizes.max().item())

        positions = torch.arange(
            self.max_group_size,
            device=group_ids.device,
        )

        # True marks padding.
        self.kp_mask = positions.unsqueeze(0) >= group_sizes.unsqueeze(1)
        self.emplacement_ids = torch.where(~self.kp_mask)

        # Sort using dense IDs such as [0, 0, 1], not raw IDs [2, 2, 5].
        self.instance_sorter = torch.argsort(
            dense_group_ids,
            stable=True,
        )
        self.inverse_instance_sorter = torch.argsort(
            self.instance_sorter,
            stable=True,
        )


        self.kp_mask = torch.zeros(self.n_groups, self.max_group_size, dtype=torch.bool)  # this mask can also be prefilled
        for gid, gs in zip(unique_group_ids, group_sizes):
            # embs[gid, :gs] = tree_preds[group_ids == gid]
            self.kp_mask[gid, gs:] = True
        self.emplacement_ids = tuple(torch.argwhere(~self.kp_mask).T)
        self.instance_sorter = torch.argsort(group_ids)
        self.inverse_instance_sorter = torch.argsort(self.instance_sorter)

    def _raw_attention_logits(self, query, embs):
        """
        Rebuild pre-softmax attention logits from the current Q/K projections.

        PyTorch MultiheadAttention does not expose raw logits, only normalized
        weights. We reconstruct them here so the ranking loss can supervise the
        exact quantity used before softmax.
        """
        if not self.attention._qkv_same_embed_dim:
            raise ValueError('Raw attention ranking expects query/key/value to share embed_dim')

        q_weight, k_weight, _ = self.attention.in_proj_weight.chunk(3, dim=0)
        if self.attention.in_proj_bias is None:
            q_bias = k_bias = None
        else:
            q_bias, k_bias, _ = self.attention.in_proj_bias.chunk(3, dim=0)

        q = torch.nn.functional.linear(query, q_weight, q_bias)
        # print(q.shape)
        k = torch.nn.functional.linear(embs, k_weight, k_bias)
        # print(k.shape)
        batch_size, target_len, embed_dim = q.shape
        source_len = k.shape[1]
        num_heads = self.attention.num_heads
        head_dim = embed_dim // num_heads

        q = q.reshape(batch_size, target_len, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch_size, source_len, num_heads, head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        # each head has its own softmax, has its own instance score
        logits = logits.squeeze(2)
        return logits

    def forward(self, tree_preds, group_ids):

        group_ids = group_ids.reshape(-1).to(
            device=tree_preds.device,
            dtype=torch.long,
        )

        if len(group_ids) != len(tree_preds):
            raise ValueError(
                "group_ids must contain one ID per instance"
        )

        self._recompute_group_cache(group_ids)
        embed_dim = tree_preds.shape[1]
        # Cache the dense tree embeddings that are actually consumed by the
        # attention layer in the current GradBoostingClassifier pathway.
        self.last_instance_embeddings = tree_preds.detach().clone()

        # Pack flattened instance embeddings into a padded bag-major tensor.
        embs = torch.zeros(self.n_groups, self.max_group_size, embed_dim, dtype=tree_preds.dtype)

        embs[self.emplacement_ids] = tree_preds[self.instance_sorter]
        
        query = self.query.expand(embs.shape[0], 1, embs.shape[2])

        # Keep raw logits in flattened instance order so ranking loss and
        # downstream inspection use the same per-instance convention.
        # shape [n_bags, max_bag_size, n_heads]
        padded_attention_logits = self._raw_attention_logits(query, embs)

        # move the instance dimension before the head dimension 
        # [n_bags, max_bag_size, n_heads]
        bag_instance_head_logits=padded_attention_logits.permute(0,2,1)

        # select only readl instances, excluding padding 
        bag_ids, positions=self.emplacement_ids

        # instances are currently ordered by group/bag
        # shape [n_instances, n_heads]
        grouped_per_head_logits=bag_instance_head_logits[bag_ids, positions]

        # restore the original flattened instance order
        per_head_logits=grouped_per_head_logits[self.inverse_instance_sorter]

        # do not detach this tensor, rank loss needs its gradient 
        self.last_attention_logits_per_head=per_head_logits

        # attention_logits = torch.empty(len(tree_preds), 1, dtype=tree_preds.dtype, device=tree_preds.device)
        # attention_logits[self.instance_sorter] = padded_attention_logits[self.emplacement_ids].reshape((-1, 1))
        self.last_attention_logits = per_head_logits.mean(
            dim=1,
            keepdim=True
        )

        group_embeddings, attention_weights = self.attention(
            query,
            embs,
            embs,
            key_padding_mask=self.kp_mask,
            is_causal=False,
        )

        # Store normalized attention weights in the same flattened order as the
        # original instances. This makes inference-time inspection easy.
        attention_weights = attention_weights.squeeze(1)
        flat_attention_weights = torch.empty(len(tree_preds), 1, dtype=tree_preds.dtype, device=tree_preds.device)
        flat_attention_weights[self.instance_sorter] = attention_weights[self.emplacement_ids].reshape((-1, 1))
        self.last_attention_weights = flat_attention_weights
        group_embeddings = group_embeddings.squeeze(1)
        return self.linear(group_embeddings)

class SetTreeNN(TreeNN):
    def __post_init__(self):
        self.history = defaultdict(list)
        self.enable_postiter_nn = False
        self.nn_lr = 1.e-4
        self.nn_steps = 1
        self.nn_num_heads = 4
        self.dropout = 0.0
        self.random_state = 1
        self.loss_fn = 'se'
        self.rank_loss_weight = 0.0
        self.rank_loss_margin = 0.0
        self.instance_loss_weight = 0.0
        self.instance_labels = None
        self.instance_labels_torch_ = None
        torch.manual_seed(self.random_state)
        self.metrics = {
            'r2': r2_score,
        }
        self._rank_pair_cache=None
        self.make_nn = lambda: (
            AttentionAggregationNN(
                embed_dim=self.embedding_size,
                num_heads=self.nn_num_heads,
                dropout=self.dropout,
                out_features=self.n_outputs_,
            )
        )
    
    def set_instance_labels(self, instance_labels):
        self.instance_labels=instance_labels
        self._rank_pair_cache=None
        return self

    def set_embedding_size(self, embedding_size: int):
        self.embedding_size = embedding_size
        return self

    def set_nn_lr(self, nn_lr: float):
        self.nn_lr = nn_lr
        return self

    def set_nn_steps(self, nn_steps: int):
        self.nn_steps = nn_steps
        return self

    def set_nn_num_heads(self, nn_num_heads: int):
        self.nn_num_heads = nn_num_heads
        return self

    def set_dropout(self, dropout: float):
        self.dropout = dropout
        return self

    def set_loss_fn(self, loss_fn: str):
        self.loss_fn = loss_fn
        return self

    def set_rank_loss(self, instance_labels=None, rank_loss_weight: float = 0.0,
                      rank_loss_margin: float = 0.0, instance_loss_weight: float = 0.0):
        self.rank_loss_weight = rank_loss_weight
        if not 0<=rank_loss_weight<=1:
            raise ValueError("rank_loss_weight must be between 0 and 1")
        self.rank_loss_margin = rank_loss_margin
        self.instance_loss_weight = instance_loss_weight
        if instance_labels is not None:
            self.set_instance_labels(instance_labels)
        return self

    def set_make_nn(self, make_nn):
        self.make_nn = make_nn

    def _postiter_nn(self, X_torch, y_torch, cumulative_predictions,
                     eval_X_nn=None,
                     eval_y=None,
                     eval_cumulative_predictions=None):
        if not self.enable_postiter_nn:
            return
        with torch.inference_mode():
            preds = self._predict_nn(X_torch, cumulative_predictions)
            self.history['loss/train'].append(
                self.__loss_fn(X_torch, y_torch, preds).item()
            )
            for name, metric_fn in self.metrics.items():
                self.history[name + '/train'].append(
                    metric_fn(y_torch.numpy(), preds.numpy())
                )
            if eval_cumulative_predictions is not None:
                assert eval_y is not None
                eval_preds = self._predict_nn(eval_X_nn, eval_cumulative_predictions)
                self.history['loss/val'].append(
                    self.__loss_fn(eval_X_nn, eval_y, eval_preds).item()
                )
                for name, metric_fn in self.metrics.items():
                    self.history[name + '/val'].append(
                        metric_fn(eval_y.numpy(), eval_preds.numpy())
                    )

    def _pretrain_nn(self, X_nn_torch, y_torch):
        self.n_outputs_ = y_torch.shape[1]
        self.nn_ = self.make_nn().to(torch.float64)
        self.optim_ = torch.optim.AdamW(self.nn_.parameters(), lr=self.nn_lr)
        self._rank_pair_cache=None
        if self.instance_labels is not None:
            self.instance_labels_torch_ = torch.as_tensor(
                self.instance_labels,
                dtype=y_torch.dtype,
            )
        else:
            self.instance_labels_torch_ = None

    def _predict_nn(self, cur_X_torch, cur_trees_predictions_torch):
        return self.nn_(cur_trees_predictions_torch, group_ids=cur_X_torch)

    def _predict_attention_outputs(self, cur_X_torch, cur_trees_predictions_torch):
        """Run a forward pass and return the cached per-instance attention data."""
        with torch.inference_mode():
            self._predict_nn(cur_X_torch, cur_trees_predictions_torch)
            return (
                self.nn_.last_instance_embeddings.detach().clone(),
                self.nn_.last_attention_logits.detach().clone(),
                self.nn_.last_attention_weights.detach().clone(),
            )

    def predict_instance_embeddings(self, X, X_nn):
        """Return flattened dense per-instance tree embeddings used by attention."""
        with torch.inference_mode():
            self.predict(X=X, X_nn=X_nn)
            return self.nn_.last_instance_embeddings.detach().clone()

    def predict_attention_logits(self, X, X_nn):
        """Return flattened raw attention logits for the provided MIL batch."""
        with torch.inference_mode():
            self.predict(X=X, X_nn=X_nn)
            return self.nn_.last_attention_logits.detach().clone()

    def predict_attention_weights(self, X, X_nn):
        """Return flattened post-softmax attention weights for the MIL batch."""
        with torch.inference_mode():
            self.predict(X=X, X_nn=X_nn)
            return self.nn_.last_attention_weights.detach().clone()

    def __aligned_instance_labels(self, scores, cur_X_torch):
        if self.instance_labels_torch_ is None:
            return None
        if len(self.instance_labels_torch_) != len(cur_X_torch):
            return None

        labels = self.instance_labels_torch_.to(device=scores.device, dtype=scores.dtype)
        if labels.ndim == 1:
            labels = labels.reshape((-1, 1))
        if labels.shape[1] == 1 and scores.shape[1] != 1:
            labels = labels.expand((-1, scores.shape[1]))
        if labels.shape != scores.shape:
            raise ValueError(
                f'instance_labels shape {tuple(labels.shape)} is incompatible with '
                f'attention logits shape {tuple(scores.shape)}'
            )
        return labels

    @staticmethod
    def __valid_instance_label_mask(labels):
        return torch.isfinite(labels) & (labels >= 0.0)
        
    def __build_rank_pair_cache(
        self,
        scores,
        labels,
        group_ids,
    ):
        """
        Precompute all valid higher-label/lower-label instance pairs.

        Each stored pair receives a weight that preserves the original
        calculation:

            sum over bags(
                mean over valid heads(
                    mean over valid pairs(loss)
                )
            )
        """
        positive_indices = []
        negative_indices = []
        head_indices = []
        pair_weights = []

        for gid in torch.unique(group_ids):
            bag_indices = torch.nonzero(
                group_ids == gid,
                as_tuple=False,
            ).flatten()

            bag_labels = labels[bag_indices]
            valid_labels = self.__valid_instance_label_mask(
                bag_labels
            )

            bag_head_pairs = []

            for head_id in range(scores.shape[1]):
                head_valid = valid_labels[:, head_id]

                if int(head_valid.sum()) < 2:
                    continue

                valid_global_indices = bag_indices[head_valid]
                valid_y = bag_labels[
                    head_valid,
                    head_id,
                ]

                # pair_i should rank above pair_j.
                pair_i, pair_j = torch.where(
                    valid_y[:, None] > valid_y[None, :]
                )

                if pair_i.numel() == 0:
                    continue

                bag_head_pairs.append(
                    (
                        head_id,
                        valid_global_indices[pair_i],
                        valid_global_indices[pair_j],
                    )
                )

            if not bag_head_pairs:
                continue

            # The original implementation first averages heads within
            # each bag and then sums the bag losses.
            n_valid_heads = len(bag_head_pairs)

            for (
                head_id,
                positive_index,
                negative_index,
            ) in bag_head_pairs:
                n_pairs = positive_index.numel()

                positive_indices.append(positive_index)
                negative_indices.append(negative_index)

                head_indices.append(
                    torch.full(
                        (n_pairs,),
                        head_id,
                        dtype=torch.long,
                        device=scores.device,
                    )
                )

                pair_weights.append(
                    torch.full(
                        (n_pairs,),
                        1.0 / (n_valid_heads * n_pairs),
                        dtype=scores.dtype,
                        device=scores.device,
                    )
                )

        if positive_indices:
            positive_indices = torch.cat(positive_indices)
            negative_indices = torch.cat(negative_indices)
            head_indices = torch.cat(head_indices)
            pair_weights = torch.cat(pair_weights)
        else:
            positive_indices = torch.empty(
                0,
                dtype=torch.long,
                device=scores.device,
            )
            negative_indices = torch.empty(
                0,
                dtype=torch.long,
                device=scores.device,
            )
            head_indices = torch.empty(
                0,
                dtype=torch.long,
                device=scores.device,
            )
            pair_weights = torch.empty(
                0,
                dtype=scores.dtype,
                device=scores.device,
            )

        return {
            # Retain a copy to verify that subsequent calls use the same
            # bag arrangement before reusing the cached pair indices.
            "group_ids": group_ids.detach().clone(),
            "n_heads": scores.shape[1],
            "positive_indices": positive_indices,
            "negative_indices": negative_indices,
            "head_indices": head_indices,
            "pair_weights": pair_weights,
        }

# This changes repeated ranking-loss work from:

# Scan all instances separately for every bag
# Rebuild all label-difference matrices
# Rebuild all pair masks
# Loop over every head

# to:

# Check that bag membership is unchanged
# Retrieve the already-prepared pairs
# Calculate score differences only for valid pairs
# Apply softplus once to all pairs

    def __rank_loss(self, cur_X_torch):
        """Pairwise rank loss on raw attention logits inside each bag and each attention logit

        Only annotated positive-negative pairs contribute. Within each bag we
        average over available pairs so a densely annotated bag does not
        dominate the ranking signal.
        """
        scores = getattr(self.nn_, 'last_attention_logits_per_head', None)
        if scores is None:
            return None
        labels = self.__aligned_instance_labels(scores, cur_X_torch)
        if labels is None:
            return None

        group_ids = cur_X_torch.ravel().to(device=scores.device, dtype=torch.long)

        cache = self._rank_pair_cache

        cache_is_valid = (
            cache is not None
            and cache["n_heads"] == scores.shape[1]
            and cache["group_ids"].shape == group_ids.shape
            and torch.equal(
                cache["group_ids"],
                group_ids,
            )
        )

        if not cache_is_valid:
            cache = self.__build_rank_pair_cache(
                scores,
                labels,
                group_ids,
            )
            self._rank_pair_cache = cache

        positive_indices = cache["positive_indices"]

        if positive_indices.numel() == 0:
            return scores.new_zeros(())

        negative_indices = cache["negative_indices"]
        head_indices = cache["head_indices"]
        pair_weights = cache["pair_weights"]

        score_differences = (
            scores[
                positive_indices,
                head_indices,
            ]
            - scores[
                negative_indices,
                head_indices,
            ]
            - self.rank_loss_margin
        )

        pair_losses = torch.nn.functional.softplus(
            -score_differences
        )

        return torch.sum(
            pair_weights * pair_losses
        )

    def __instance_loss(self, cur_X_torch):
        scores = getattr(self.nn_, 'last_attention_logits', None)
        if scores is None:
            return None
        labels = self.__aligned_instance_labels(scores, cur_X_torch)
        if labels is None:
            return None

        valid_labels = self.__valid_instance_label_mask(labels)
        if not torch.any(valid_labels):
            return scores.new_zeros(())
        
        # annotated positive instances should have high raw attnetion logits, annotated negative instances should have low raw attention logits 
        return torch.nn.functional.binary_cross_entropy_with_logits(
            scores[valid_labels],
            labels[valid_labels],
            reduction='sum',
        )

    def __add_auxiliary_losses(self, bag_loss, cur_X_torch):
        """Combine bag, ranking, and instance losses as a convex mixture."""

        rank_weight = self.rank_loss_weight
        instance_weight = self.instance_loss_weight
        bag_weight = 1.0 - rank_weight - instance_weight

        if rank_weight < 0.0 or instance_weight < 0.0:
            raise ValueError("Loss weights must be non-negative")

        if bag_weight < 0.0:
            raise ValueError(
                "rank_loss_weight + instance_loss_weight must be <= 1"
            )

        loss = bag_weight * bag_loss

        if rank_weight:
            rank_loss = self.__rank_loss(cur_X_torch)
            if rank_loss is None:
                raise RuntimeError(
                    "Ranking loss is enabled but could not be calculated"
                )
            loss = loss + rank_weight * rank_loss

        if instance_weight:
            instance_loss = self.__instance_loss(cur_X_torch)
            if instance_loss is None:
                raise RuntimeError(
                    "Instance loss is enabled but could not be calculated"
                )
            loss = loss + instance_weight * instance_loss

        return loss

    def __loss_fn(self, cur_X_torch, cur_y_torch, nn_preds):
        # group_ids = cur_X_torch
        if callable(self.loss_fn):
            bag_loss = self.loss_fn(nn_preds, cur_y_torch)
        elif self.loss_fn.lower() == 'se':
            bag_loss = (cur_y_torch - nn_preds).pow(2).sum()
        elif self.loss_fn.lower() == 'bce':
            bag_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                nn_preds,
                cur_y_torch,
                reduction='sum'
            )
        else:
            raise ValueError(f'Wrong {self.loss_fn=!r}')
        return self.__add_auxiliary_losses(bag_loss, cur_X_torch)

    def _post_update_nn(self, X_nn_torch, y_torch, sample_ids_torch, cumulative_predictions):
        for _ in range(self.nn_steps):
            self.optim_.zero_grad()
            nn_preds = self._predict_nn(X_nn_torch, cumulative_predictions)
            loss = self.__loss_fn(X_nn_torch, y_torch, nn_preds)
            loss.backward()
            self.optim_.step()

    def _calc_sample_grads(self, cur_X_torch, cur_y_torch,
                           cur_trees_predictions_torch,
                           cur_sample_predictions):
        nn_preds = self._predict_nn(cur_X_torch, cur_trees_predictions_torch)
        loss = self.__loss_fn(cur_X_torch, cur_y_torch, nn_preds)
        grads, = torch.autograd.grad(loss, cur_sample_predictions)
        return grads
