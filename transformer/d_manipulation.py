import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from itertools import chain, combinations


class ModifyVisitsToNumeric:
    def __init__(self, mode='A', columns=None):
        if columns is None:
            columns = ['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']
        self.mode = mode
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        dic = {}
        if self.mode == 'A':
            dic = {'never': 0, 'less1': 1, '1~3': 2, '4~8': 6, 'gt8': 12}
        for column in self.columns:
            X[column].replace(dic, inplace=True)
            X[column] = X[column].astype('int')
        return X


class ClusterCatAndSetDummies:
    def __init__(self, mode='Hierarchical', n_clusters=3, columns=None):
        if columns is None:
            columns = ['education', 'occupation', 'income']
        self.mode = mode
        self.n_clusters = n_clusters
        self.columns = columns
        self.labels_dict = dict.fromkeys(columns, {})

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        for column in self.columns:
            if self.mode == 'Hierarchical':
                cluster_df = data.groupby([column, 'coupon']).agg({'Y': 'mean'}).unstack()
                cluster_df.columns = cluster_df.columns.map(lambda par: par[1])

                hc = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='euclidean', linkage='ward')
                hc = hc.fit(cluster_df)
                # cluster_labels = [column + '_' + str(val) for val in hc.labels_]
                dummy_dict = {}
                for key, val in zip(cluster_df.index, hc.labels_):
                    dummy_dict[key] = val
                self.labels_dict[column] = dummy_dict
            if self.mode == 'Kmean':
                cluster_df = data.groupby([column, 'coupon']).agg({'Y': 'mean'}).unstack()
                cluster_df.columns = cluster_df.columns.map(lambda par: par[1])

                kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(cluster_df)
                dummy_dict = {}
                for key, val in zip(cluster_df.index, kmeans.labels_):
                    dummy_dict[key] = val
                self.labels_dict[column] = dummy_dict
        return self

    def transform(self, X):
        for column in self.columns:
            X = X.replace({column: self.labels_dict[column]})
        X = pd.get_dummies(X, columns=self.columns)
        return X


class TargetEncoder:
    def __init__(self, columns=None, mode='All'):
        if mode == 'All':
            columns = ['passenger', 'temperature', 'time', 'coupon', 'expiration', 'gender',
                       'age', 'maritalStatus', 'has_children', 'education', 'occupation',
                       'income', 'Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20',
                       'Restaurant20To50', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min', 'direction_same']
        if mode == 'combination':
            columns = ['education', 'occupation', 'income', 'coupon']
        if (mode == 'custom') and (columns is not None):
            columns = columns

        self.columns = columns
        self.mode = mode
        self.labels_dict = dict.fromkeys(columns, {})

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        if (self.mode == 'All') or (self.mode == 'custom'):
            # data['temperature'] = data['temperature'].astype('str')
            for column in self.columns:
                column_df = data.groupby(column).agg({'Y': 'mean'})
                self.labels_dict[column] = column_df

        if self.mode == 'combination':
            for comb in combinations(self.columns, 2):
                cluster_df = data.groupby([comb[0], comb[1]]).agg({'Y': 'mean'}).unstack()
                cluster_df.columns = cluster_df.columns.map(lambda par: par[1])
                self.labels_dict[comb[0] + '@' + comb[1]] = cluster_df
            for column in self.columns:
                column_df = data.groupby(column).agg({'Y': 'mean'})
                self.labels_dict[column] = column_df
        return self

    def transform(self, X):
        if self.mode == 'All' or self.mode == 'custom':
            for column in self.columns:
                X[column] = X[column].map(lambda parm: self.labels_dict[column].loc[parm, 'Y'])

        if self.mode == 'combination':
            for comb in combinations(self.columns, 2):
                X[comb[0] + '@' + comb[1]] = X[comb[0]] + '@' + X[comb[1]]
                X[comb[0] + '@' + comb[1]] = X[comb[0] + '@' + comb[1]].map(
                    lambda parm: self.labels_dict[comb[0] + '@' + comb[1]].loc[parm.split('@')[0], parm.split('@')[1]])
            for column in self.columns:
                X[column] = X[column].map(lambda parm: self.labels_dict[column].loc[parm, 'Y'])
        return X


class ModifyAgeToNumeric:
    def __init__(self, mode='A', columns=None):
        if columns is None:
            columns = ['age']
        self.mode = mode
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        dic = {}
        if self.mode == 'A':
            dic = {'below21': 17, '21': 21, '26': 26, '31': 31, '36': 36, '41': 41, '46': 46, '50plus': 60}
        for column in self.columns:
            X[column].replace(dic, inplace=True)
            X[column] = X[column].astype('int')
        return X


class ModifyHourToNumeric:
    def __init__(self, mode='A', columns=None):
        if columns is None:
            columns = ['time']
        self.mode = mode
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        dic = {}
        if self.mode == 'A':
            for column in self.columns:
                X[column] = X[column].map(
                    lambda par: par.replace('AM', '') if 'AM' in par else int(par.replace('PM', '')) + 12)
                X[column] = X[column].astype('int')
        return X


class ModifyToDummies:
    def __init__(self, mode='A', columns=None):
        if columns is None:
            columns = ['destination', 'passanger', 'weather', 'coupon', 'maritalStatus']
        self.mode = mode
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.mode == 'A':
            X = pd.get_dummies(X, columns=self.columns)
        return X


class ModifyToBinary:
    def __init__(self, mode='A', columns=None):
        if columns is None:
            columns = ['expiration', 'gender']
        self.mode = mode
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.mode == 'A':
            X = pd.get_dummies(X, columns=self.columns, drop_first=True)
        # print(X.info())
        return X
