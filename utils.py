import pandas as pd
# -------- classification
import sklearn
from sklearn import neighbors, tree, ensemble, naive_bayes, svm
# *** KNN
from sklearn.neighbors import KNeighborsClassifier
# *** Decision Tree; Random Forest
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# *** Naive Bayes
from sklearn.naive_bayes import GaussianNB
# *** SVM classifier
from sklearn.svm import SVC
# --------  metrics:
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import make_scorer


def get_df_data():
    df = pd.read_csv('./data/in-vehicle-coupon-recommendation.csv')

    # "car" has only 108 values
    df.drop(labels=['car'], inplace=True, axis=1)
    df.dropna(inplace=True)

    # "toCoupon_GEQ5min" is fixed to 1
    df.drop(labels=['toCoupon_GEQ5min'], inplace=True, axis=1)

    # "direction_opp" and "direction_same" are opposites
    df.drop(labels=['direction_opp'], inplace=True, axis=1)

    df.replace({"coupon": {'Restaurant(<20)': 'Restaurant(-20)'}}, inplace=True)
    return df


def get_classifier_obj(classifier_name, params=None):
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
