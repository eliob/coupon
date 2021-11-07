import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn

import transformer.models as mdl
import transformer.d_manipulation as d_mnp
import utils

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = utils.get_df_data()
    # print(sorted(sklearn.metrics.SCORERS.keys()))

    for classifier in [
        # ('SVM', SVC()),
        ('CatBoostAgg_model', CatBoostClassifier(iterations=1000, verbose=200, task_type='CPU')),
        ('xgb_model', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='auc')),
        ('Knn_model', KNeighborsClassifier(n_neighbors=9)),
        ('RandomForest_model', RandomForestClassifier(n_estimators=51)),
        ('DecisionTree_model', DecisionTreeClassifier(max_depth=6, min_samples_split=20)),
        ('CatBoost_model', CatBoostClassifier(iterations=1000, verbose=200, task_type='CPU',
                                              cat_features=['destination', 'occupation', 'education', 'income',
                                                            'passanger', 'weather', 'coupon',
                                                            'maritalStatus', 'expiration', 'gender', 'Bar',
                                                            'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20',
                                                            'Restaurant20To50', 'age', 'time'])),
    ]:
        print('-------------------------', classifier[0], '-----------------------------------')
        # for cluster_mode in ['Hierarchical', 'Kmean']:
        #     for occ_cluster in [3, 4]:
        #         for other_cluster in [2, 3]:
        X = df.drop(labels=['Y'], axis=1)
        y = df.Y

        # preprocess
        # t = d_mnp.ClusterCatAndSetDummies(mode=cluster_mode, columns=['occupation'], n_clusters=occ_cluster)
        # t.fit(X, y)
        # X = t.transform(X)
        #
        # t = d_mnp.ClusterCatAndSetDummies(mode=cluster_mode, columns=['education', 'income'],
        #                                   n_clusters=other_cluster)
        # t.fit(X, y)
        # X = t.transform(X)

        if classifier[0] != 'CatBoost_model':
            t = d_mnp.TargetEncoder(columns=['education', 'occupation', 'income', 'coupon'])
            t.fit(X, y)
            X = t.transform(X)

            t = d_mnp.ModifyToDummies(mode='A',
                                      columns=['destination', 'passanger', 'weather', 'maritalStatus'])
            X = t.transform(X)

            t = d_mnp.ModifyToBinary(mode='A', columns=['expiration', 'gender'])
            X = t.transform(X)

        if classifier[0] != 'CatBoost_model':
            steps = [
                ('visits', d_mnp.ModifyVisitsToNumeric(mode='A',
                                                       columns=['Bar', 'CoffeeHouse', 'CarryAway',
                                                                'RestaurantLessThan20', 'Restaurant20To50'])),
                ('age', d_mnp.ModifyAgeToNumeric(mode='A', columns=['age'])),
                ('time', d_mnp.ModifyHourToNumeric(mode='A', columns=['time'])),
                #  last one should be the model !
                # ('Model', mdl.LastTransformer())
                classifier]
        else:
            steps = [classifier]

        coupon_pipeline = Pipeline(steps)
        my_cv = StratifiedShuffleSplit(n_splits=8, train_size=0.75, test_size=0.25)
        # scoring: f1, neg_log_loss, error, recall
        scores = cross_val_score(coupon_pipeline, X, y, cv=my_cv, scoring='roc_auc')
        print('Scores:', round(np.mean(scores), 4), [round(x, 4) for x in scores])

        t = d_mnp.ModifyVisitsToNumeric(mode='A', columns=['Bar', 'CoffeeHouse', 'CarryAway',
                                                           'RestaurantLessThan20', 'Restaurant20To50'])
        X = t.transform(X)
        t = d_mnp.ModifyAgeToNumeric(mode='A', columns=['age'])
        X = t.transform(X)
        t = d_mnp.ModifyHourToNumeric(mode='A', columns=['time'])
        X = t.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        coupon_clf = classifier[1].fit(X_train, y_train)

        print(utils.report(coupon_clf, X_test, y_test))
