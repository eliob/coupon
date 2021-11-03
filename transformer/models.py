

class LastTransformer():
    def __init__(self):
        self.y = []
        pass

    def fit(self, X, y=None):
        print('LastTransformer - fit')
        self.y = y
        return self

    def predict(self, X):
        print(X)
        print(X.info())
        return self.y


