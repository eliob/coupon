import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas_profiling
import matplotlib.pyplot as plt


def get_df_data():
    df = pd.read_csv('./data/in-vehicle-coupon-recommendation.csv')
    # profile = df.profile_report(title="Coupon Profiling Report", correlations=None)
    # profile.to_file("profile_row_data.html")

    # "car" has only 108 values
    df.drop(labels=['car'], inplace=True, axis=1)
    df = df.dropna().reset_index(drop=True)

    # "toCoupon_GEQ5min" is fixed to 1
    df.drop(labels=['toCoupon_GEQ5min'], inplace=True, axis=1)

    # profile = df.profile_report(title="Coupon Profiling Report")
    # profile.to_file("profile_row_data2.html")

    # "direction_opp" and "direction_same" are opposites
    # "temperature" and "weather" have high correlations
    df.drop(labels=['destination', 'direction_opp', 'weather'], inplace=True, axis=1)

    df.replace({"coupon": {'Restaurant(<20)': 'Restaurant(-20)'}}, inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True, ignore_index=True)

    # profile = df.profile_report(title="Coupon Profiling Report")
    # profile.to_file("profile_data_eng_data.html")
    return df


def split_to_train_and_test(dataset, label_column, test_ratio, rand_state):
    X = dataset.drop(label_column, axis='columns')
    y = dataset[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=rand_state)
    return X_train, X_test, y_train, y_test


def report(clf, X, y):
    acc = accuracy_score(y_true=y, y_pred=clf.predict(X))
    prec = precision_score(y_true=y, y_pred=clf.predict(X))
    recall = recall_score(y_true=y, y_pred=clf.predict(X))
    f1 = f1_score(y_true=y, y_pred=clf.predict(X))
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])

    # cm = pd.DataFrame(confusion_matrix(y_true=y, y_pred=clf.predict(X)), index=clf.classes_, columns=clf.classes_)
    # rep = classification_report(y_true=y, y_pred=clf.predict(X))

    return {'Accuracy': acc, 'Prec.': prec, 'Recall': recall, 'F1': f1, 'AUC': auc}


def calc_evaluation_val(eval_metric, y_test, y_predicted):
    if eval_metric == 'accuracy':
        return metrics.accuracy_score(y_true=y_test, y_pred=y_predicted)

    if eval_metric == 'precision':
        return metrics.precision_score(y_true=y_test, y_pred=y_predicted)

    if eval_metric == 'recall':
        return metrics.recall_score(y_true=y_test, y_pred=y_predicted)

    if eval_metric == 'f1':
        return metrics.f1_score(y_true=y_test, y_pred=y_predicted)

    if eval_metric == 'confusion_matrix':
        return metrics.confusion_matrix(y_true=y_test, y_pred=y_predicted)


def my_pipeline(X, y, steps, mode='transform'):
    for step in steps:
        transformer = step[1]
        if mode == 'fit_transform':
            X = transformer.fit(X, y).transform(X)
        else:
            X = transformer.transform(X)
    return X


def build_plot(coupon_clf, X_test, y_test, name):
    scores = coupon_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, scores, pos_label=1)
    plot_auc = roc_auc_score(y_test, coupon_clf.predict_proba(X_test)[:, 1]) * 100
    plt.plot(fpr, tpr, '-', marker=".", markersize=4, label=f'({plot_auc:2.3f}%) {name}')


def show_plot():
    plt.title('ROC')
    plt.xlabel('FPR (False Positive Rate = 1-specificity)')
    plt.ylabel('TPR (True Positive Rate = sensitivity)')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
