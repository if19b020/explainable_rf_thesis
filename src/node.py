class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, class_counts=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left # child node
        self.right = right # child node
        self.value = value # only for leaf nodes
        self.class_counts = class_counts # for feature contribution
        
    def is_leaf_node(self):
        return self.value is not None