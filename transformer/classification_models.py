from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree

# class LastTransformer():
#     def __init__(self):
#         self.y = []
#         pass
#
#     def fit(self, X, y=None):
#         print('LastTransformer - fit')
#         self.y = y
#         return self
#
#     def predict(self, X):
#         print(X)
#         print(X.info())
#         return self.y


def get_classifier_obj(classifier_name, params):
    if classifier_name == 'KNN':
        if params is not None:
            return KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        else:
            return KNeighborsClassifier()

    if classifier_name == 'decision_tree':
        if params is not None:
            return DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'])
        else:
            return DecisionTreeClassifier()

    if classifier_name == 'random_forest':
        if params is not None:
            return RandomForestClassifier(n_estimators=params['n_estimators'])
        else:
            return RandomForestClassifier()

    if classifier_name == 'naive_bayes':
        return GaussianNB()

    if classifier_name == 'svm':
        return SVC()


def find_best_k_for_KNN(X_train, y_train):
    parameters = {'n_neighbors': [3, 7, 9, 11, 25]}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters,scoring=make_scorer(metrics.f1_score, greater_is_better=True))
    clf.fit(X_train, y_train)
    best_K = clf.best_params_['n_neighbors']
    best_f1_val = clf.best_score_
    return best_K, best_f1_val


def find_best_decision_tree_params(X_train, y_train):
    parameters = {'max_depth': [2, 4, 6, 8, 10], "min_samples_split": [5, 10, 20, 40, 80]}
    dt = tree.DecisionTreeClassifier()
    clf = GridSearchCV(dt, parameters, scoring=make_scorer(metrics.f1_score, greater_is_better=True))
    clf.fit(X_train, y_train)
    best_max_depth = clf.best_params_['max_depth']
    best_min_samples_split = clf.best_params_['min_samples_split']
    best_f1_val = clf.best_score_
    return best_max_depth, best_min_samples_split, best_f1_val


def find_best_random_forest_num_estimators(X_train, y_train):
    parameters = {'n_estimators': [11, 51, 71, 91]}
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters, scoring=make_scorer(metrics.f1_score, greater_is_better=True))
    clf.fit(X_train, y_train)
    best_num_estimators = clf.best_params_['n_estimators']
    best_f1_val = clf.best_score_
    return best_num_estimators, best_f1_val




