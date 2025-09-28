class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, class_counts=None):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left 
        self.right_child = right
        self.value = value
        self.class_counts = class_counts
        
    def is_leaf_node(self):
        return self.value is not None