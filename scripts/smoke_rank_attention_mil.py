import numpy as np

from sit.gradtree.grad_boosting import GradBoostingClassifier


def make_tiny_mil_data(random_state=0):
    rng = np.random.default_rng(random_state)

    # Number of instances in each bag. MILData uses this to rebuild bag
    # boundaries from the flattened instance table X.
    group_sizes = np.array([4, 5, 3, 6, 4, 5])

    # We first build bags independently, then flatten them into X.
    bags = []
    bag_y = []
    instance_y = []

    for bag_id, bag_size in enumerate(group_sizes):
        X_b = rng.normal(size=(bag_size, 3))
        if bag_id % 2 == 0:
            # Positive bags: seed one clearly positive-looking instance and one
            # clearly negative-looking instance so rank loss has a supervised pair.
            X_b[0] += np.array([3.0, 2.0, 0.0])
            X_b[-1] += np.array([-2.5, -1.5, 0.0])
            y_b = 1
            inst_b = np.full(bag_size, np.nan)
            inst_b[0] = 1.0
            inst_b[-1] = 0.0
        else:
            # Negative bags still matter for bag BCE. We shift the features so
            # they are easier to separate, but keep instance labels unknown.
            X_b += np.array([-1.5, -1.0, 0.0])
            y_b = 0
            inst_b = np.full(bag_size, 0.0)

        bags.append(X_b)
        bag_y.append(y_b)
        instance_y.append(inst_b)

    X = np.concatenate(bags, axis=0).astype(np.float64)
    y = np.asarray(bag_y, dtype=np.float64)
    instance_y = np.concatenate(instance_y, axis=0).astype(np.float64)
    return X, y, group_sizes, instance_y


def main():
    X, y, group_sizes, instance_y = make_tiny_mil_data()

    # Small settings keep the smoke test fast while still exercising the full
    # bag-loss + attention-ranking training path.
    model = GradBoostingClassifier(
        lam_2=0.001,
        lr=1.0,
        max_depth=1,
        splitter="random",
        n_estimators=1, 
        n_update_iterations=1,
        embedding_size=4,
        nn_lr=1e-4,
        nn_num_heads=1,
        nn_steps=1,
        dropout=0.0,
        lambda_rank=0.1,
        rank_margin=0.0,
        lambda_inst=0.0,
    )

    model.fit(X, y, group_sizes, instance_y=instance_y)

    # This is the dense per-instance tree embedding consumed by attention in the
    # current GradBoostingClassifier pathway. Its last dimension should match
    # embedding_size regardless of tree depth.
    instance_embeddings = model.predict_instance_embeddings(X, group_sizes)
    print(type(instance_embeddings))          
    print(len(instance_embeddings))           

    print(type(instance_embeddings[0]))      
    print(instance_embeddings[0].shape)       
    print(instance_embeddings[1].shape)       
    print(instance_embeddings[1])     
    print(instance_embeddings[0].dtype)       
    print(instance_embeddings[0].ndim)        
    print(instance_embeddings[0].size)        
    # This is a different inspection view: a bag-level histogram over visited
    # leaves for one tree. It is useful for understanding routing, but it is
    # not the dense embedding used by attention.

if __name__ == "__main__":
    main()
