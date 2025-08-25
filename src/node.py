class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left # child node
        self.right = right # child node
        self.value = value # only for leaf nodes
        
    def is_leaf_node(self):
        return self.value is not None