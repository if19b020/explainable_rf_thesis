from decision_tree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=5, max_depth=5, min_samples_split=2, n_features=5, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state
        self.trees: list[DecisionTree] = []
        self.paths = []
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                         n_features=int(np.sqrt(X.shape[1])))
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict_and_export(self, X, y=None, feature_names=None, filename="paths.json"):
        """
        Predict X and export paths and feature contributions to JSON file.
        """
        import json
        all_data = []
        y_pred = []

        for sample_idx, x in enumerate(X):
            sample_paths = self._trace_paths(x, feature_names=feature_names)
            pred_class = int(self.predict([x])[0])
            y_pred.append(pred_class)

            # Get each tree's prediction
            tree_preds = [tree.predict([x])[0] for tree in self.trees]
            vote_share = tree_preds.count(pred_class) / len(tree_preds)

            entry = {
                "sample_index": sample_idx,
                "predicted_class": pred_class
            }

            if y is not None:
                entry["true_class"] = int(y[sample_idx])

            entry["vote_share"] = round(vote_share, 2)  # which percentage of trees voted for the class

            entry["feature_contributions"] = self._compute_feature_contributions(x, feature_names)

            entry["paths"] = [
                {
                    "tree": tree_idx,
                    "tree_prediction": int(tree_preds[tree_idx]),
                    "steps": steps
                }
                for tree_idx, steps in sample_paths
            ]
            
            all_data.append(entry)

        with open(filename, "w") as f:
            json.dump(all_data, f, indent=2)
            
        return y_pred
        
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([self._most_common_label(pred) for pred in predictions.T])
            
    def _trace_paths(self, x, feature_names=None):
        """
        Trace decision paths for all trees in the forest.
        """

        all_paths = []
        for i, tree in enumerate(self.trees):
            steps = tree.trace_path(x, feature_names=feature_names)
            all_paths.append((i, steps))
        return all_paths
            
    def _compute_feature_contributions(self, x, feature_names=None):
        """
        Compute average feature contributions across all trees for a given sample.
        Returns:
            {
                "base_value": float,
                "final_value": float,
                "contributions": {feature_name: avg_contribution}
            }
        """
        # Get the predicted class from the forest
        target_class = int(self.predict([x])[0])

        total_contributions = {}
        base_values = []
        final_values = []

        for tree in self.trees:
            base, final, contribs = tree.compute_contributions(x, target_class, feature_names)
            base_values.append(base)
            final_values.append(final)

            # Aggregate contributions
            for feat, val in contribs.items():
                total_contributions[feat] = total_contributions.get(feat, 0) + val

        # Average over all trees
        avg_contributions = {k: v / len(self.trees) for k, v in total_contributions.items()}
        sorted_contributions = dict(
            sorted(
                ((k, round(v, 3)) for k, v in avg_contributions.items()),
                key=lambda item: item[1],  # sort by value
                reverse=True
            )
        )

        return {
            "base_value":       round( sum(base_values) / len(base_values), 3 ),
            "final_value":      round( sum(final_values) / len(final_values), 3 ),
            "contributions":    sorted_contributions
        }

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True,)
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common