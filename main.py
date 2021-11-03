import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
        ('SVM', SVC()),
        ('xgb_model', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')),
        ('Knn_model', KNeighborsClassifier(n_neighbors=9)),
        ('RandomForest_model', RandomForestClassifier(n_estimators=51)),
        ('DecisionTree_model', DecisionTreeClassifier(max_depth=6, min_samples_split=20))
    ]:
        print('-------------------------', classifier[0], '-----------------------------------')
        for cluster_mode in ['Hierarchical', 'Kmean']:
            for occ_cluster in [3, 4]:
                for other_cluster in [2, 3]:

                    steps = [
                        ('visits', d_mnp.ModifyVisitsToNumeric(mode='A',
                                                               columns=['Bar', 'CoffeeHouse', 'CarryAway',
                                                                        'RestaurantLessThan20', 'Restaurant20To50'])),
                        ('age', d_mnp.ModifyAgeToNumeric(mode='A', columns=['age'])),
                        ('time', d_mnp.ModifyHourToNumeric(mode='A', columns=['time'])),
                        #  last one should be the model !
                        # ('Model', mdl.LastTransformer())
                        classifier]

                    coupon_pipeline = Pipeline(steps)

                X = df.drop(labels=['Y'], axis=1)
                y = df.Y

                # preprocess
                t = d_mnp.ClusterCatAndSetDummies(mode=cluster_mode, columns=['occupation'], n_clusters=occ_cluster)
                t.fit(X, y)
                X = t.transform(X)

                t = d_mnp.ClusterCatAndSetDummies(mode=cluster_mode, columns=['education', 'income'],
                                                  n_clusters=other_cluster)
                t.fit(X, y)
                X = t.transform(X)

                t = d_mnp.ModifyToDummies(mode='A',
                                          columns=['destination', 'passanger', 'weather', 'coupon', 'maritalStatus'])
                X = t.transform(X)

                t = d_mnp.ModifyToBinary(mode='A', columns=['expiration', 'gender'])
                X = t.transform(X)

                my_cv = StratifiedShuffleSplit(n_splits=6, train_size=0.7, test_size=0.3)
                # scoring: f1, neg_log_loss, error, recall
                scores = cross_val_score(coupon_pipeline, X, y, cv=my_cv, scoring='roc_auc')
                print(f'{cluster_mode:20} {occ_cluster} {other_cluster}', 'Scores:', round(np.mean(scores), 4), scores)

    # print(X)
    # print(X.info())
