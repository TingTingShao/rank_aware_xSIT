import torch
from ..mil.data import MILData
from gradient_growing_trees.tree_nn import TreeNN
from gradient_growing_trees.tree import BatchArbitraryLoss
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone, defaultdict
from sklearn.metrics import r2_score
from abc import ABCMeta, abstractmethod


class AttentionAggregationNN(torch.nn.Module):

    # add the bag feature dim
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, out_features: int = 1, bag_feature_dim: int = 0):
        super().__init__()
        self.bag_feature_dim = bag_feature_dim

        self.attention = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # query is a learnable parameter that is used to compute the attention logits
        self.query = torch.nn.Parameter(torch.ones((1, 1, embed_dim), dtype=torch.float32, requires_grad=True))

        # a linear head -> final prediction of the model

        # embed_dim + bag feature dim: input feature dim of the linear layer
        # out_features: output feature dim of the linear layer, one dim for the linear head to generate the final prediction
        self.linear=torch.nn.Linear(embed_dim+bag_feature_dim, out_features)
        # self.linear = torch.nn.Linear(embed_dim, out_features)

        self.group_ids = None
        self.last_instance_embeddings = None
        self.last_attention_logits = None
        self.last_attention_weights = None
        self.last_group_embeddings = None

    def _recompute_group_cache(self, group_ids):
        if group_ids is self.group_ids:
            return
        # print('Recomputing cache')
        self.group_ids = group_ids

        group_ids = group_ids.ravel().to(torch.long)
        # These operations can be run once in advance
        unique_group_ids, group_sizes = torch.unique(group_ids, return_counts=True)
        self.n_groups = len(unique_group_ids)
        self.max_group_size = torch.max(group_sizes)

        # this is to mark valid instance positions inside each bag/group
        # [number of groups, maximum number of instances per group]
        # example
        # bag 0 has 3 embryos
        # bag 1 has 1 embryo
        # bag 2 has 4 embryos
#      tensor([
#     [False, False, False, False],
#     [False, False, False, False],
#     [False, False, False, False]
# ])
        self.kp_mask = torch.zeros(self.n_groups, self.max_group_size, dtype=torch.bool, device=group_ids.device)  # this mask can also be prefilled

        for gid, gs in zip(unique_group_ids, group_sizes):
            # embs[gid, :gs] = tree_preds[group_ids == gid]
            self.kp_mask[gid, gs:] = True
        self.emplacement_ids = tuple(torch.argwhere(~self.kp_mask).T)
        self.instance_sorter = torch.argsort(group_ids)

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
        # store one score per instance, so multi-head logits are averaged.
        logits = logits.squeeze(2).mean(dim=1)
        return logits

    def _split_group_context(self, group_context):
        if group_context.ndim==1 or group_context.shape[1]==1:
            return group_context.reshape((-1, 1)), None
        
        group_ids=group_context[:, :1]
        repeated_bag_features = group_context[:, 1:]
        return group_ids, repeated_bag_features
    
    def _bag_features_by_group(self, group_ids, repeated_bag_features):
        if repeated_bag_features is None:
            if self.bag_features_dim: 
                raise ValueError("Bag features are required for this model")
            return None

        if repeated_bag_features.shape[1]!=self.bag_feature_dim:
            raise ValueError("Bag features must have dimension %d, got %d" % (self.bag_feature_dim, repeated_bag_features.shape[1]))
        
        flat_group_ids=group_ids.ravel().to(torch.long)
        unique_group_ids=torch.unique(flat_group_ids, sorted=True)

        expected_group_ids=torch.arange(
            len(unique_group_ids),
            device=flat_group_ids.device,
            dtype=flat_group_ids.dtype,
        )

        if not torch.equal(unique_group_ids, expected_group_ids):
            raise ValueError("Group ids must be contiguous integers starting from 0")
        
        bag_features=[]
        for grid in unique_group_ids:
            cur=repeated_bag_features[flat_group_ids==grid]

            if cur.shape[0]==0:
                raise ValueError("Group %d has no instances" % grid)

            if not torch.allclose(cur, cur[:1].expand_as(cur), equal_nan=True):
                raise ValueError("Group %d has inconsistent bag features" % grid)

            bag_features.append(cur[0])
        
        return torch.stack(bag_features, dim=0)


    
    def forward(self, tree_preds, group_context):

        group_ids, repeated_bag_features=self._split_group_context(group_context)
        self._recompute_group_cache(group_ids)

        embed_dim = tree_preds.shape[1]
        # Cache the dense tree embeddings that are actually consumed by the
        # attention layer in the current GradBoostingClassifier pathway.
        self.last_instance_embeddings = tree_preds.detach().clone()

        # Pack flattened instance embeddings into a padded bag-major tensor.
        embs = torch.zeros(
            self.n_groups, 
            self.max_group_size, 
            embed_dim, 
            dtype=tree_preds.dtype,
            device=tree_preds.device)

        embs[self.emplacement_ids] = tree_preds[self.instance_sorter]
        query = self.query.expand(embs.shape[0], 1, embs.shape[2])

        # Keep raw logits in flattened instance order so ranking loss and
        # downstream inspection use the same per-instance convention.
        padded_attention_logits = self._raw_attention_logits(query, embs)
        attention_logits = torch.empty(
            len(tree_preds), 
            1, 
            dtype=tree_preds.dtype, 
            device=tree_preds.device)

        attention_logits[self.instance_sorter] = padded_attention_logits[self.emplacement_ids].reshape((-1, 1))
        self.last_attention_logits = attention_logits

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
        flat_attention_weights = torch.empty(
            len(tree_preds), 
            1, 
            dtype=tree_preds.dtype, 
            device=tree_preds.device)

        flat_attention_weights[self.instance_sorter] = attention_weights[self.emplacement_ids].reshape((-1, 1))
        self.last_attention_weights = flat_attention_weights

        group_embeddings = group_embeddings.squeeze(1)
        self.last_group_embedding=group_embeddings.detach().clone()

        bag_features=self._bag_features_by_group(group_ids, repeated_bag_features)

        if bag_features is not None:
            group_embeddings = torch.cat([group_embeddings, bag_features], dim=1)

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
        self.make_nn = lambda: (
            AttentionAggregationNN(
                embed_dim=self.embedding_size,
                num_heads=self.nn_num_heads,
                dropout=self.dropout,
                out_features=self.n_outputs_,
            )
        )

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
        self.rank_loss_margin = rank_loss_margin
        self.instance_loss_weight = instance_loss_weight
        if instance_labels is not None:
            self.set_instance_labels(instance_labels)
        return self

    def set_instance_labels(self, instance_labels):
        self.instance_labels = instance_labels
        return self

    def set_make_nn(self, make_nn):
        # self.make_nn = make_nn
        self.make_nn=lambda: (
            AttentionAggregationNN(
                embed_dim=self.embedding_size, 
                num_heads=self.nn_num_heads,
                dropout=self.dropout,
                out_features=self.n_outputs_,
                bag_feature_dim=self.bag_feature_dim,
            )
        )
    
    def set_bag_feature_dim(self, bag_feature_dim:int):
        self.bag_feature_dim=bag_feature_dim
        return self

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

    def __rank_loss(self, cur_X_torch):
        """Pairwise rank loss on raw attention logits inside each bag.

        Only annotated positive-negative pairs contribute. Within each bag we
        average over available pairs so a densely annotated bag does not
        dominate the ranking signal.
        """
        scores = getattr(self.nn_, 'last_attention_logits', None)
        if scores is None:
            return None
        labels = self.__aligned_instance_labels(scores, cur_X_torch)
        if labels is None:
            return None

        group_ids = cur_X_torch.ravel().to(device=scores.device, dtype=torch.long)
        losses = []
        for gid in torch.unique(group_ids):
            group_mask = group_ids == gid
            group_scores = scores[group_mask]
            group_labels = labels[group_mask]
            valid_labels = self.__valid_instance_label_mask(group_labels)
            for output_id in range(group_scores.shape[1]):
                output_valid = valid_labels[:, output_id]
                output_labels = group_labels[:, output_id]
                # embryo label [0, 1] -> 0: negative, 1: positive, inbetween: soft label when the number of embryos transferred (n) is smaller than the number of live birth (m) m/n
                # output_label: calssifies the embryos with soft labels as negative !!!!
                # "improvement": make use of the soft labels 
                pos_scores = group_scores[output_valid & (output_labels > 0.5), output_id]
                neg_scores = group_scores[output_valid & (output_labels <= 0.5), output_id]

                if len(pos_scores) == 0 or len(neg_scores) == 0:
                    continue
                # panelize cases where positive attention logits are not higher than negative attention logist by a margin
                # diffs = pos_scores[:, None] - neg_scores[None, :] - self.rank_loss_margin
                # losses.append(torch.nn.functional.softplus(-diffs).mean())

                # example
                # label 1.0 should rank above label 0.67
                # label 0.67 should rank above label 0.33
                # label 0.33 should rank above label 0.0
                valid_scores=group_scores[output_valid, output_id]
                valid_y=output_labels[output_valid]

                # # pairwise label difference 
                label_diff=valid_y[:, None] - valid_y[None, :]

                # # embryo i should rank above embryo j if lable_i > label_j
                pair_mask=label_diff>0

                if pair_mask.sum()==0:
                    continue 

                score_diff=valid_scores[:, None] - valid_scores[None, :]

                diffs=score_diff[pair_mask]-self.rank_loss_margin

                pair_losses=torch.nn.functional.softplus(-diffs)

                losses.append(pair_losses.mean())

        if len(losses) == 0:
            return scores.new_zeros(())
        return torch.stack(losses).sum()

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
        """Attach optional rank / instance supervision to the bag loss."""
        loss = bag_loss
        if self.rank_loss_weight:
            rank_loss = self.__rank_loss(cur_X_torch)
            if rank_loss is not None:
                loss = loss + self.rank_loss_weight * rank_loss
        if self.instance_loss_weight:
            instance_loss = self.__instance_loss(cur_X_torch)
            if instance_loss is not None:
                loss = loss + self.instance_loss_weight * instance_loss
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
