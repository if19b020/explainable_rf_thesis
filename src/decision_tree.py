import numpy as np
from node import Node

class DecisionTree:
    def __init__(self, max_depth = 5, min_samples_split = 3, n_features = 5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features # number of features considered for the tree
        self.root = None
        self.predictions = []
        self.paths = []
    
    def fit(self, X, y):
        """Train the decision tree using recursive splitting."""
        X = np.array(X)
        y = np.array(y)
        self.n_features = min(self.n_features, X.shape[1]) 

        self.root = self._grow_tree(X, y, depth=0)        

    def predict(self, X):
        self.paths = []  # Reset paths on every prediction
        predictions = []
        X = np.array(X) 

        for sample in X:
            path = []
            prediction = self._traverse_tree(self.root, sample, path)
            self.paths.append(path)
            predictions.append(prediction)

        return np.array(predictions)
    
    def trace_path(self, x, feature_names=None, pretty_print=False):
        """
        Trace the decision path for a single sample.

        Parameters
        ----------
        x : array-like
            Feature values for a single sample.
        feature_names : list of str, optional
            Names of the features; if None, generic names are used.
        pretty_print : bool, optional
            If True, prints an ASCII-style tree path.

        Returns
        -------
        path : list of dict
            Each dict contains: feature, threshold, direction, value.
        """
        path = []
        node = self.root

        while node is not None and node.feature is not None:
            feature_idx = node.feature
            feature_name = (
                feature_names[feature_idx] if feature_names is not None else f"Feature[{feature_idx}]"
            )

            value = x[feature_idx]
            direction = "<=" if value <= node.threshold else ">"
            path.append({
                'feature': feature_name,
                'threshold': node.threshold,
                'direction': direction,
                'value': value
            })
            node = node.left if value <= node.threshold else node.right

        if pretty_print:
            print("Decision Path:")
            for step_num, step in enumerate(path, start=1):
                print(
                    f"  Step {step_num}: {step['feature']} = {step['value']:.2f} "
                    f"{step['direction']} {step['threshold']:.2f}"
                )

        return path
    
        
    def _grow_tree(self, X, y, depth):
        """Recursively grows the decision tree."""
        # Stopping condition: Max depth reached or too few samples
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return self._create_leaf(y)

        # Find best feature and threshold for splitting
        feature, threshold = self._find_best_split(X, y)
        
        if feature is None:  # If no valid split found, return leaf
            return self._create_leaf(y)

        # Split data into left and right branches
        left_idx = X[:, feature] < threshold
        right_idx = ~left_idx

        # Recursively grow left and right branches
        left_child = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature, threshold, left=left_child, right=right_child)

    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split the data."""
        best_feature, best_threshold = None, None
        best_gain = 0  # Track highest information gain

        features = np.random.choice(X.shape[1], self.n_features, replace=False)
        for feature in features:
            thresholds = np.unique(X[:, feature])  # Unique feature values
            if len(thresholds) < 2:  # Skip constant features
                continue

            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        """Calculate information gain for a given split."""
        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        # Edge case: If all samples go to one side, IG = 0
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        # Compute impurity before and after the split
        impurity_before = self._gini_impurity(y)
        impurity_after = (
            (len(y[left_mask]) / len(y)) * self._gini_impurity(y[left_mask]) +
            (len(y[right_mask]) / len(y)) * self._gini_impurity(y[right_mask])
        )

        return impurity_before - impurity_after

    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels."""
        if len(y) == 0:  # Edge case: No data
            return 0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _create_leaf(self, y):
        """Create a leaf node by returning the majority class."""
        return Node(value=np.bincount(y).argmax())
    
    def _traverse_tree(self, node, x, path):
        """Recursively traverse the tree to make a prediction."""
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            path.append((node.feature, node.threshold, "<=", x[node.feature]))
            return self._traverse_tree(node.left, x, path)
        else:
            path.append((node.feature, node.threshold, ">", x[node.feature]))
            return self._traverse_tree(node.right, x, path)
        
    