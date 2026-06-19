import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_random_state
from gradient_growing_trees.tree import GradientGrowingTreeRegressor
from .set_tree_nn import SetTreeNN
from .embedder import leaf_indices_into_bag_embeddings
from ..mil.data import MILData


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def _group_instance_outputs(values, mil_data):
    """Convert flattened per-instance outputs back into one array per bag."""
    grouped = []
    for start, end in zip(mil_data.shifts[:-1], mil_data.shifts[1:]):
        cur = values[start:end]
        if cur.ndim == 2 and cur.shape[1] == 1:
            cur = cur[:, 0]
        grouped.append(cur)
    return grouped


def _configure_rank_loss(stnn, params, instance_y):
    """Translate public hyperparameters into SetTreeNN rank-loss settings."""
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


def _get_tree_estimator(stnn, tree_idx: int):
    """Return one fitted gradient tree by index."""
    if not hasattr(stnn, 'estimators_'):
        raise ValueError('The model is not fitted yet.')
    if tree_idx < 0 or tree_idx >= len(stnn.estimators_):
        raise IndexError(f'tree_idx={tree_idx} is out of range for {len(stnn.estimators_)} fitted trees')
    return stnn.estimators_[tree_idx]

def _validate_bag_X(bag_X, group_sizes):
    if bag_X is None:
        return None
    
    if bag_X.ndim==1:
        bag_X=bag_X.reshape(-1, 1)
    
    if bag_X.ndim!=2:
        raise ValueError('bag_X must be 1D or 2D')
    if bag_X.shape[0] != len(group_sizes):
        raise ValueError('bag_X must have the same number of rows as group_sizes')
    return bag_X

def _make_group_context(mil_data, bag_X=None):
    group_ids=mil_data.group_ids.reshape(-1, 1)
    bag_X=_validate_bag_X(bag_X, mil_data.group_sizes)
    if bag_X is None:
        return group_ids
    
    repeated_bag_X=np.repeat(bag_X, mil_data.group_sizes, axis=0)
    return np.concatenate([group_ids, repeated_bag_X], axis=1)


class GradBoostingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, **params):
        if 'loss_fn' not in params:
            params['loss_fn'] = 'bce'
        self.params = params

    def fit(self, X, y, group_sizes, instance_y=None, bag_X=None):
        mil_train = MILData(X, y, group_sizes, instance_y=instance_y)
        params = self.params

        bag_X=_validate_bag_X(bag_X, mil_train.group_sizes)
        bag_feature_dim=0 if bag_X is None else bag_X.shape[1]

        # Build the gradient-tree encoder and the neural bag aggregator, then
        # optionally attach auxiliary rank supervision on attention logits.
        self.stnn = SetTreeNN(
            base_estimator=GradientGrowingTreeRegressor(
                lam_2=params['lam_2'],
                lr=params['lr'],
                splitter=params['splitter'],
                max_depth=params['max_depth'],
                random_state=1,
            ),
            n_estimators=params['n_estimators'],
            lam_2=params['lam_2'],
            lr=params['lr'],
            tree_loss_on_sample_ids=False,
            n_update_iterations=params['n_update_iterations'],
        ).set_embedding_size(params['embedding_size'])\
         .set_nn_lr(params['nn_lr'])\
         .set_nn_num_heads(params['nn_num_heads'])\
         .set_nn_steps(params['nn_steps'])\
         .set_dropout(params['dropout'])\
         .set_bag_feature_dim(bag_feature_dim) # add bag feature dim

        if 'loss_fn' in params:
            self.stnn.set_loss_fn(params['loss_fn'])

        _configure_rank_loss(self.stnn, params, mil_train.instance_y)

        self.stnn.enable_postiter_nn = False
        self.stnn.fit(
            mil_train.X,
            mil_train.y.reshape((-1, 1)) if mil_train.y.ndim == 1 else mil_train.y,
            # X_nn=mil_train.group_ids.reshape((-1, 1)),
            X_nn=
            # eval_XyXnn=(mil_test.X, mil_test.y.reshape((-1, 1)), mil_test.group_ids.reshape((-1, 1)))
        )
        return self

    def predict_proba(self, X, group_sizes):
        mil_train = MILData(X, None, group_sizes)
        proba = sigmoid(self.stnn.predict(X=mil_train.X, X_nn=mil_train.group_ids.reshape((-1, 1))).numpy())
        return np.concatenate([1.0 - proba, proba], axis=1)

    def predict(self, X, group_sizes):
        return np.argmax(self.predict_proba(X, group_sizes), axis=1)

    def predict_attention_logits(self, X, group_sizes, grouped: bool = True):
        """Return raw attention logits, either flattened or grouped by bag."""
        mil_data = MILData(X, None, group_sizes)
        logits = self.stnn.predict_attention_logits(
            X=mil_data.X,
            X_nn=mil_data.group_ids.reshape((-1, 1)),
        ).numpy()
        return _group_instance_outputs(logits, mil_data) if grouped else logits

    def predict_attention_weights(self, X, group_sizes, grouped: bool = True):
        """Return post-softmax attention weights, either flattened or per bag."""
        mil_data = MILData(X, None, group_sizes)
        weights = self.stnn.predict_attention_weights(
            X=mil_data.X,
            X_nn=mil_data.group_ids.reshape((-1, 1)),
        ).numpy()
        return _group_instance_outputs(weights, mil_data) if grouped else weights

    def predict_instance_embeddings(self, X, group_sizes, grouped: bool = True):
        """Return the dense per-instance tree embeddings used by attention."""
        mil_data = MILData(X, None, group_sizes)
        embeddings = self.stnn.predict_instance_embeddings(
            X=mil_data.X,
            X_nn=mil_data.group_ids.reshape((-1, 1)),
        ).numpy()
        return _group_instance_outputs(embeddings, mil_data) if grouped else embeddings

    def get_tree_estimator(self, tree_idx: int = 0):
        """Expose one fitted tree so its raw attributes can be inspected."""
        return _get_tree_estimator(self.stnn, tree_idx)

    def predict_leaf_indices(self, X, group_sizes, tree_idx: int = 0, grouped: bool = True):
        """Return the leaf id reached by each instance for one fitted tree."""
        mil_data = MILData(X, None, group_sizes)
        leaf_ids = _get_tree_estimator(self.stnn, tree_idx).apply(mil_data.X)
        return _group_instance_outputs(leaf_ids, mil_data) if grouped else leaf_ids

    def predict_leaf_bag_embeddings(self, X, group_sizes, tree_idx: int = 0, normalize: bool = True):
        """Return bag-level leaf-count embeddings for one fitted tree.

        Each column corresponds to a leaf reached by at least one instance in the
        input data. With normalize=True, counts are divided by bag size.
        """
        mil_data = MILData(X, None, group_sizes)
        leaf_ids = _get_tree_estimator(self.stnn, tree_idx).apply(mil_data.X)
        bag_embeddings, all_leaves = leaf_indices_into_bag_embeddings(
            leaf_ids,
            mil_data.group_sizes,
            all_leaves=None,
            normalize=normalize,
        )
        return bag_embeddings, all_leaves


class GradBoostingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, group_sizes, instance_y=None):
        mil_train = MILData(X, y, group_sizes, instance_y=instance_y)
        params = self.params

        # Same encoder/aggregator stack as the classifier, but with a
        # regression bag loss on top instead of BCE.
        self.stnn = SetTreeNN(
            base_estimator=GradientGrowingTreeRegressor(
                lam_2=params['lam_2'],
                lr=params['lr'],
                splitter=params['splitter'],
                max_depth=params['max_depth'],
                random_state=1,
            ),
            n_estimators=params['n_estimators'],
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
            # X_nn=mil_train.group_ids.reshape((-1, 1)),
            X_nn=_make_group_context(mil_tran, bag_X),
            # eval_XyXnn=(mil_test.X, mil_test.y.reshape((-1, 1)), mil_test.group_ids.reshape((-1, 1)))
        )
        return self

    def predict(self, X, group_sizes, bag_X=None):
        # mil_test = MILData(X, None, group_sizes)
        # return self.stnn.predict(X=mil_test.X, X_nn=mil_test.group_ids.reshape((-1, 1))).numpy()
        return np.argmax(
            self.predict_prob(X, group_sizes, bag_X=bag_X),
            axis=1
        )

    def predict_proba(self, X, group_sizes, bag_X=None):
        mil_train=MILData(X, None, group_sizes)

        proba=sigmoid(
            self.stnn.predict(
                X=mil_train.X, 
                X_nn=_make_group_context(mil_train, bag_X)
            ).numpy()

        )
        return np.concatenate([1-proba, proba], axis=1)

    def predict_attention_logits(self, X, group_sizes, grouped: bool = True, bag_X=None):
        """Return raw attention logits, either flattened or grouped by bag."""
        mil_data = MILData(X, None, group_sizes)
        logits = self.stnn.predict_attention_logits(
            X=mil_data.X,
            X_nn=_make_group_context(mil_data, bag_X)).numpy()
        
        return _group_instance_outputs(logits, mil_data) if grouped else logits

    def predict_attention_weights(self, X, group_sizes, grouped: bool = True, bag_X=None):
        """Return post-softmax attention weights, either flattened or per bag."""
        mil_data = MILData(X, None, group_sizes)
        weights = self.stnn.predict_attention_weights(
            X=mil_data.X,
            X_nn=_make_group_context(mil_data, bag_X)
        ).numpy()
        return _group_instance_outputs(weights, mil_data) if grouped else weights

    def predict_instance_embeddings(self, X, group_sizes, grouped: bool = True, bag_X=None):
        """Return the dense per-instance tree embeddings used by attention."""
        mil_data = MILData(X, None, group_sizes)
        embeddings = self.stnn.predict_instance_embeddings(
            X=mil_data.X,
            X_nn=_make_group_context(mil_data, bag_X)
        ).numpy()
        return _group_instance_outputs(embeddings, mil_data) if grouped else embeddings

    def get_tree_estimator(self, tree_idx: int = 0):
        """Expose one fitted tree so its raw attributes can be inspected."""
        return _get_tree_estimator(self.stnn, tree_idx)

    def predict_leaf_indices(self, X, group_sizes, tree_idx: int = 0, grouped: bool = True):
        """Return the leaf id reached by each instance for one fitted tree."""
        mil_data = MILData(X, None, group_sizes)
        leaf_ids = _get_tree_estimator(self.stnn, tree_idx).apply(mil_data.X)
        return _group_instance_outputs(leaf_ids, mil_data) if grouped else leaf_ids

    def predict_leaf_bag_embeddings(self, X, group_sizes, tree_idx: int = 0, normalize: bool = True):
        """Return bag-level leaf-count embeddings for one fitted tree."""
        mil_data = MILData(X, None, group_sizes)
        leaf_ids = _get_tree_estimator(self.stnn, tree_idx).apply(mil_data.X)
        bag_embeddings, all_leaves = leaf_indices_into_bag_embeddings(
            leaf_ids,
            mil_data.group_sizes,
            all_leaves=None,
            normalize=normalize,
        )
        return bag_embeddings, all_leaves
