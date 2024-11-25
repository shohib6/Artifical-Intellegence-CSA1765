import numpy as np

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(y) == 0:
            return None
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return TreeNode(value=np.bincount(y).argmax())

        best_split = self._find_best_split(X, y)
        if not best_split:
            return TreeNode(value=np.bincount(y).argmax())

        X_left, y_left, X_right, y_right = self._split_data(
            X, y, best_split['feature_index'], best_split['threshold']
        )

        if len(y_left) == 0 or len(y_right) == 0:
            return TreeNode(value=np.bincount(y).argmax())

        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return TreeNode(
            feature_index=best_split['feature_index'],
            threshold=best_split['threshold'],
            left=left_subtree,
            right=right_subtree
        )

    def _find_best_split(self, X, y):
        best_split = {}
        best_gini = float('inf')

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split_data(X, y, feature_index, threshold)
                gini = self._gini_index(y_left, y_right)

                if gini < best_gini:
                    best_gini = gini
                    best_split = {'feature_index': feature_index, 'threshold': threshold}

        return best_split

    def _split_data(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        return X[left_mask], y[left_mask], X[~left_mask], y[~left_mask]

    def _gini_index(self, y_left, y_right):
        n_left, n_right = len(y_left), len(y_right)
        if n_left == 0 or n_right == 0:
            return 0

        gini_left = 1.0 - sum((np.sum(y_left == c) / n_left) ** 2 for c in np.unique(y_left))
        gini_right = 1.0 - sum((np.sum(y_right == c) / n_right) ** 2 for c in np.unique(y_right))
        return (n_left * gini_left + n_right * gini_right) / (n_left + n_right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

def create_synthetic_data():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y

def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(len(y) * test_size)
    train_indices, test_indices = indices[:-test_size], indices[-test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

if __name__ == "__main__":
    X, y = create_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")
