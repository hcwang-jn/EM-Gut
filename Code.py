import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, RepeatedKFold


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from joblib import Parallel, delayed
import pickle

# subregion select
def subregion_select(X):

    '''
    X: pandas Dataframe of subregions; index: sample_id;
       contain a list of subregion informations, column name as 'subregion';
       contain a list of age informations, column name as 'age'.
    '''

    enc = OneHotEncoder()
    onehotencoded = enc.fit_transform(X.subregion.reshape(-1,1)).toarray() # list to onehot
    col_name = [str(i) for i in enc.get_feature_names()] # get columns name
    onehotencoded = pd.DataFrame(onehotencoded, columns=col_name, index=X.index)

    y = X.age

    model = RandomForestRegressor()
    results, ranks = [], []

    for i in range(onehotencoded.shape[1]):
        model.fit(onehotencoded, y.values.ravel()) # fit subregions and age

        fs = model.feature_importances_
        fs = pd.DataFrame(fs, index=onehotencoded.columns, columns=['score']).sort_values(by='score', ascending=False)
        r2 = cross_val_score(model, onehotencoded, y.values.ravel(), cv=RepeatedKFold(n_repeats=10, n_splits=5), n_jobs=-1).mean()
        
        results.append(r2) # R2 in this epoch
        ranks.append(fs.index[0]) # the most importance feature in this epoch

        # remove the most importance feature in this epoch
        onehotencoded = onehotencoded.loc[:, fs.index[1:]].loc[onehotencoded.sum(axis=1) != 0, :]
        y = y.reindex(onehotencoded.index)
    
    return results, ranks
    # results: R2 of each epoch 
    # ranks: Subregion factors in descending order of influence on age


# model based feature selection method
class EmbeddingRegressionSelector(object):
    def __init__(self, model):
        self.model = model

    def select(self, X, y):
        fs = self.model.fit(X, y).feature_importances_
        fs_col = pd.DataFrame(fs, columns=['score'])
        
        self.sel_idx = np.array(fs_col.index)
        selected = X[:, self.sel_idx]
        return selected
    
    def fidx(self):
        return self.sel_idx

# Model ensemble
class StackedEnsembleRegressor(object):
    def __init__(self, models, cv=RepeatedKFold(n_repeats=1, n_splits=5), emethod=LinearRegression()):
        self.cv = cv
        self.models = models
        self.ensemble_method = emethod
    
    def fit(self, Xs, y):
        if type(Xs) != list:
            Xs = [Xs]

        cv_idxs = [[a,b] for a,b in self.cv.split(Xs[0])]
        xs_preds, self.xs_models = [],[]
        for X in Xs:
            m_lev = []
            for model in self.models:
                preds, ms = [], []
                for cv_idx in cv_idxs:
                    Xtrain = X[cv_idx[0], :]
                    Xvalid = X[cv_idx[1], :]
                    ytrain = y[cv_idx[0]]

                    model.fit(Xtrain, ytrain)
                    pred = model.predict(Xvalid)
                    preds.append(pred)

                    msave = pickle.dumps(model)
                    ms.append(msave)

                preds = np.concatenate(preds).reshape(-1,1)
                xs_preds.append(preds)

                m_lev.append(ms)
            
            self.xs_models.append(m_lev)
        xs_preds = np.concatenate(xs_preds, axis=1)
        
        y_sort = [i[1] for i in cv_idxs]
        y_sort = np.concatenate(y_sort)
        y_sort = y[y_sort]

        self.ensemble_method.fit(xs_preds, y_sort)
    
    def predict(self, Xs):
        if type(Xs) != list:
            Xs = [Xs]
        
        xs_preds = []
        for ms,X in zip(self.xs_models, Xs):
            for m in ms:
                preds = [pickle.loads(m_ind).predict(X).reshape(-1,1) for m_ind in m]
                preds = np.concatenate(preds, axis=1).mean(axis=1).reshape(-1,1)
                xs_preds.append(preds)
        
        xs_preds = np.concatenate(xs_preds, axis=1)
        pred = self.ensemble_method.predict(xs_preds)

        return pred

# Permutation feature importance
def PFI(X1, X2, y, cv_idx):
    '''
        X1: input datasets, in this study is species;
        X2: input datasets, in this study is pathways;
        y: metadata include age.
        cv_idx: cross_validation index [[a,b] for a,b in cv.split(data7)]
    '''
    y = y.age.values.ravel()

    models = [] # models used in model ensemble 
    # E.g.
    # models = [Lasso(), ElasticNet(), BayesianRidge(), SVR(), RandomForestRegressor(), GradientBoostingRegressor(),
    #       XGBRegressor(), XGBRFRegressor(), LGBMRegressor()]

    stacker = StackedEnsembleRegressor(models)
    
    Xtrainsp = X1[cv_idx[0], :]
    Xtrainpw = X2[cv_idx[0], :]
    Xtrain = [Xtrainsp, Xtrainpw]
    ytrain = y[cv_idx[0]]
    
    Xtestsp = X1[cv_idx[1], :]
    Xtestpw = X2[cv_idx[1], :]
    Xtest = [Xtestsp, Xtestpw]
    ytest = y[cv_idx[1]]
    
    stacker.fit(Xtrain, ytrain)
    pred_base = stacker.predict(Xtest)
    
    r2 = [r2_score(ytest, pred_base)]
    lab = ['base']
    
    for n in range(Xtestsp.shape[1]-1):
        zero = Xtestsp.copy()
        zero[:, n] = 0
        
        pred_zero = stacker.predict([zero, Xtestpw])
        
        r2.append(r2_score(ytest, pred_zero))
        lab.append('SP_F_' + str(n))
    
    for n in range(Xtestpw.shape[1]-1):
        zero = Xtestpw.copy()
        zero[:, n] = 0
        
        pred_zero = stacker.predict([Xtestsp, zero])
        
        r2.append(r2_score(ytest, pred_zero))
        lab.append('PW_F_' + str(n))
    
    return [r2, lab]