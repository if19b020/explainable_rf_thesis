from decision_tree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        self.paths = []
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        np.random.seed(12)

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                         n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def trace_paths(self, x, feature_names=None, pretty_print=False):
        """
        Trace decision paths for all trees in the forest.

        Parameters
        ----------
        x : array-like
            Feature values for a single sample.
        feature_names : list of str, optional
        pretty_print : bool, optional

        Returns
        -------
        all_paths : list
            List of (tree_index, path) pairs.
        """

        all_paths = []
        for i, tree in enumerate(self.trees):
            steps = tree.trace_path(x, feature_names=feature_names, pretty_print=pretty_print)
            all_paths.append((i, steps))
        return all_paths
    
    def save_paths(self, X, y=None, feature_names=None, filename="paths.json"):
        """
        Save decision paths and predictions for multiple samples to a JSON file.

        Parameters
        ----------
        X : array-like
            Dataset to trace.
        y : array-like, optional
            True labels (for reference).
        feature_names : list of str, optional
            Feature names for readability.
        filename : str
            Path to output JSON file.
        """
        import json
        all_data = []

        for sample_idx, x in enumerate(X):
            sample_paths = self.trace_paths(x, feature_names=feature_names, pretty_print=False)
            pred_class = int(self.predict([x])[0])

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
            
    def compute_feature_contributions(self, x, feature_names=None):
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

        return {
            "base_value": sum(base_values) / len(base_values),
            "final_value": sum(final_values) / len(final_values),
            "contributions": avg_contributions
        }


    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
        
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([self._most_common_label(pred) for pred in predictions.T])