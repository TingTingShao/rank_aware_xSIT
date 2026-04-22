import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..mil.data import MILData
from .set_tree_nn import SetTreeNN
from gradient_growing_trees.tree import GradientGrowingTreeRegressor


def leaf_indices_into_bag_embeddings(leaf_ids, group_sizes, all_leaves=None, normalize: bool = True):
    if all_leaves is None:
        all_leaves = np.unique(leaf_ids)
    embeddings = np.zeros((len(group_sizes), np.max(all_leaves) + 1), dtype=np.float64)
    shifts = np.insert(np.cumsum(group_sizes), 0, 0)
    for k, (i, j) in enumerate(zip(shifts[:-1], shifts[1:])):
        v = np.zeros(embeddings.shape[1])
        np.add.at(v, leaf_ids[i:j], 1)
        if normalize:
            embeddings[k] += v / (j - i)
        else:
            embeddings[k] += v
    return embeddings[:, all_leaves], all_leaves


def _configure_rank_loss(stnn, params, instance_y):
    rank_loss_weight = params.get('rank_loss_weight', params.get('lambda_rank', 0.0))
    instance_loss_weight = params.get('instance_loss_weight', params.get('lambda_inst', 0.0))
    rank_loss_margin = params.get('rank_loss_margin', params.get('rank_margin', 0.0))
    if (rank_loss_weight or instance_loss_weight) and instance_y is None:
        raise ValueError('instance_y is required when rank or instance loss is enabled')
    if instance_y is not None or rank_loss_weight or instance_loss_weight:
        stnn.set_rank_loss(
            instance_labels=instance_y,
            rank_loss_weight=rank_loss_weight,
            rank_loss_margin=rank_loss_margin,
            instance_loss_weight=instance_loss_weight,
        )


class GradientSetInputTreeEmbedder(TransformerMixin, BaseEstimator):
    def __init__(self, grad_tree_params: dict):
        self.normalize = grad_tree_params.pop('normalize', True)
        self.grad_tree_params = grad_tree_params

    def fit_transform(self, X, y, group_sizes, instance_y=None):
        mil_train = MILData(X, y, group_sizes, instance_y=instance_y)
        params = self.grad_tree_params

        self.stnn = SetTreeNN(
            base_estimator=GradientGrowingTreeRegressor(
                lam_2=params['lam_2'],
                lr=params['lr'],
                splitter=params['splitter'],
                max_depth=params['max_depth'],
                random_state=1,
            ),
            n_estimators=1,  # params['n_estimators'],
            lam_2=params['lam_2'],
            lr=params['lr'],
            tree_loss_on_sample_ids=False,
            n_update_iterations=params['n_update_iterations'],
        ).set_embedding_size(params['embedding_size'])\
         .set_nn_lr(params['nn_lr'])\
         .set_nn_num_heads(params['nn_num_heads'])\
         .set_nn_steps(params['nn_steps'])\
         .set_dropout(params['dropout'])
        if 'loss_fn' in params:
            self.stnn.set_loss_fn(params['loss_fn'])
        _configure_rank_loss(self.stnn, params, mil_train.instance_y)

        self.stnn.enable_postiter_nn = False
        self.stnn.fit(
            mil_train.X,
            mil_train.y.reshape((-1, 1)) if mil_train.y.ndim == 1 else mil_train.y,
            X_nn=mil_train.group_ids.reshape((-1, 1)),
            # eval_XyXnn=(mil_test.X, mil_test.y.reshape((-1, 1)), mil_test.group_ids.reshape((-1, 1)))
        )

        leaves = self.stnn.estimators_[0].apply(mil_train.X)
        bag_embeddings, self.all_leaves = leaf_indices_into_bag_embeddings(
            leaves,
            mil_train.group_sizes,
            all_leaves=None,
            normalize=self.normalize,
        )
        return bag_embeddings

    def transform(self, X, group_sizes):
        leaves = self.stnn.estimators_[0].apply(X)
        bag_embeddings, _ = leaf_indices_into_bag_embeddings(
            leaves,
            group_sizes,
            all_leaves=self.all_leaves,
            normalize=self.normalize,
        )
        return bag_embeddings
