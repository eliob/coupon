import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


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


def report(clf, X, y):
    acc = accuracy_score(y_true=y,
                         y_pred=clf.predict(X))
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    cm = pd.DataFrame(confusion_matrix(y_true=y,
                                       y_pred=clf.predict(X)),
                      index=clf.classes_,
                      columns=clf.classes_)
    rep = classification_report(y_true=y,
                                y_pred=clf.predict(X))
    return 'accuracy: {:.3f}\n\n{}\n\n{}\nauc: {}'.format(acc, cm, rep, auc)
