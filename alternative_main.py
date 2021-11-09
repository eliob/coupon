import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV

import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

import transformer.classification_models as mdl
import transformer.d_manipulation as d_mnp
import utils

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    df = utils.get_df_data()
    X = df.drop(labels=['Y'], axis=1)
    y = df.Y

    steps = [('target_encoder', d_mnp.TargetEncoder(columns=['education', 'occupation', 'income', 'coupon'])),
             ('modify_to_dummies',
              d_mnp.ModifyToDummies(mode='A', columns=['destination', 'passanger', 'weather', 'maritalStatus'])),
             ('modify_to_binary', d_mnp.ModifyToBinary(mode='A', columns=['expiration', 'gender'])),
             ('modify_visits', d_mnp.ModifyVisitsToNumeric(mode='A', columns=['Bar', 'CoffeeHouse', 'CarryAway',
                                                                              'RestaurantLessThan20',
                                                                              'Restaurant20To50'])),
             ('modify_age', d_mnp.ModifyAgeToNumeric(mode='A', columns=['age'])),
             ('modify_time', d_mnp.ModifyHourToNumeric(mode='A', columns=['time']))]

    X = utils.my_pipeline(X, y, steps, mode='fit_transform')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # HyperParameter Tuning
    best_K, best_f1_val = mdl.find_best_k_for_KNN(X_train, y_train)
    print(f'The best k is: {best_K}\nThe best f1 score is: {best_f1_val}')

    best_max_depth, best_min_samples_split, best_f1_val = mdl.find_best_decision_tree_params(X_train, y_train)
    print(
        f'The max depth is: {best_max_depth}\nThe best min sample is: {best_min_samples_split}\nThe best f1 score is: {best_f1_val}')

    best_num_estimators, best_f1_val = mdl.find_best_random_forest_num_estimators(X_train, y_train)
    print(f'The num estimator is: {best_num_estimators}\nThe best f1 score is: {best_f1_val}')

    # Run different classification models for comparison
    compare_model_dict = {}

    for classifier in [
        ('CatBoostAgg_model',
         CatBoostClassifier(iterations=1000, verbose=200, task_type='CPU', eval_metric='AUC', random_state=42)),
        ('xgb_model',
         xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='auc', random_state=42)),
        ('RandomForest_model', RandomForestClassifier(n_estimators=best_num_estimators)),
        ('DecisionTree_model',
         DecisionTreeClassifier(max_depth=best_max_depth, min_samples_split=best_min_samples_split)),
        ('Knn_model', KNeighborsClassifier(n_neighbors=best_K)),
        ('NaiveBayes_model', GaussianNB()),
        ('MLP_model', MLPClassifier(random_state=42, max_iter=1000)),
        ('SVM_model', SVC(kernel='linear', probability=True)),
        ('CatBoost_model', CatBoostClassifier(iterations=1000, verbose=200, task_type='CPU',
                                              eval_metric='AUC', random_state=42,
                                              cat_features=['destination', 'occupation', 'education',
                                                            'income', 'passanger', 'weather', 'coupon',
                                                            'maritalStatus',
                                                            'expiration', 'gender'])),
    ]:
        print('-------------------------', classifier[0], '-----------------------------------')

        if classifier[0] == 'CatBoost_model':
            X = df.drop(labels=['Y'], axis=1)
            y = df.Y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            steps_cat = [
                ('modify_visits', d_mnp.ModifyVisitsToNumeric(mode='A', columns=['Bar', 'CoffeeHouse', 'CarryAway',
                                                                                 'RestaurantLessThan20',
                                                                                 'Restaurant20To50'])),
                ('modify_age', d_mnp.ModifyAgeToNumeric(mode='A', columns=['age'])),
                ('modify_time', d_mnp.ModifyHourToNumeric(mode='A', columns=['time']))]
            X_train = utils.my_pipeline(X_train, y_train, steps_cat, mode='fit_transform')
            X_test = utils.my_pipeline(X_test, y_test, steps_cat, mode='transform')

        coupon_clf = classifier[1].fit(X_train, y_train)

        compare_model_dict[classifier[0]] = utils.report(coupon_clf, X_test, y_test)
        utils.build_plot(coupon_clf, X_test, y_test, classifier[0])

    compare_model_df = pd.DataFrame.from_dict(compare_model_dict).transpose().sort_values(by='AUC', ascending=False)
    compare_model_df.to_csv('compare_models_df.csv')
    filename = 'compare_models_df'
    outfile = open(filename, 'wb')
    pickle.dump(compare_model_df, outfile)
    outfile.close()
    print(compare_model_df)

    utils.show_plot()
