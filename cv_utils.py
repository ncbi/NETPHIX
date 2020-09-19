from sklearn.svm import SVR
import sklearn.ensemble as ensemble
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score, KFold, RepeatedKFold

import scipy.stats as ss
from scipy.special import erf
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, make_scorer

# ------------------------------------------------------------------------------------------------
# Cumulative distribution function
def pcidx_cdf(u,v,s):
    return 0.5 * (1 + erf(abs(u-v) / (2*s)))

# ------------------------------------------------------------------------------------------------
# Probabilistic C-Index 
# Python version of pci code originally written in R by Michael Patrick Menden
# E  : experimental values (AUC)
# s	 : standard deviation for E.
# P : the predicted rank of E. 
def comp_pci(E,s,P):
    runningSum = 0
    n = 0.5 * len(E) * (len(E)-1)
    for i in range(len(E)-1):
        for j in range(i+1, len(E)):
            if ((E[i]>E[j]) and (P[i]>P[j])) or ((E[i]<E[j]) and (P[i]<P[j])):
                runningSum += pcidx_cdf(E[i], E[j], s)
            elif ((E[i]>E[j])and (P[i]<P[j])) or ((E[i]<E[j]) and (P[i]>P[j])):
                runningSum += (1 - pcidx_cdf(E[i], E[j], s))
            else:
                runningSum += 0.5      
    return runningSum/n


def pc_idx(y_true, y_pred):
    s = np.std(y_true)
    p_rank = pd.Series(y_pred).rank()
    return comp_pci(y_true, s, p_rank)


def spearman_coef(x, y):
    return ss.spearmanr(x, y, nan_policy='omit')[0]


def assign_score_param(score):
    
    # score options
    if score == "ev":
        score_param = "explained_variance"
    elif score == "spearman":
        score_param = make_scorer(spearman_coef)
    elif score == "pci":
        score_param = make_scorer(pc_idx)
    else:
        score_param = None
        print("score function is not valid")
    return score_param


def create_model(method, score, cv = 4, n_jobs=10):
    svr_parameters = { "C":(10000, 1000, 100, 10, 1, 0.1), "epsilon":(0.2, 0.1, 0.01, 0.001, 0.0001),
                                   "gamma": (5e-7, 5e-6, 1.08e-5, 2.32e-5, 5e-5, 1.08e-4, 2.32e-4, 5e-4,5e-3, 5e-2)}
    rfr_parameters = {'n_estimators': [10, 100],
                        'max_depth': [None, 10, 100], 'min_samples_split': [1, 2, 3]}
    
    if method == "svr":
        svr_model = SVR(kernel='rbf')
        search = GridSearchCV(svr_model, cv = cv, scoring=score, param_grid=svr_parameters, n_jobs=n_jobs, verbose=1)
    elif method == "rfr":
        rfr = ensemble.RandomForestRegressor()
        search = GridSearchCV(rfr, rfr_parameters, cv = cv, scoring=score, n_jobs=n_jobs, verbose=1)
    return search

def ctrp_drug_name(drug):
    if drug.endswith("(1)") or drug.endswith("(2)") or drug.endswith("(-)"):
        drug = drug[:-4]
    return drug


# Learning hyperparameters with GDSC
def fit_gdsc_model(target_df, alt_df, module,  method, score_param, cv):
    # drug response (y)
    target_df = target_df.dropna(axis=1)
    common_columns = target_df.columns.intersection(alt_df.columns)
    y = target_df.loc[:, common_columns].T
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    y = imp_mean.fit_transform(y).ravel()
    print(target_df.shape)

    # gene alteration (X)
    X = alt_df.loc[module, common_columns].T
    imp_zero = SimpleImputer(strategy='constant', fill_value=0)
    X = imp_zero.fit_transform(X)

    model = create_model(method, score=score_param, cv=cv, n_jobs=10)
    model.fit(X, y)
    return model

# Compute CTRP score with learned model
def comp_ctrp_score(model, ctrp_auc_df, ctrp_alt_df, drug, module):
    # ctrp data
    ctrp_drug = ctrp_drug_name(drug)
    ctrp_auc = ctrp_auc_df.loc[ctrp_drug].dropna()
    ctrp_common_columns = ctrp_auc.index.intersection(ctrp_alt_df.columns)

    imp_zero = SimpleImputer(strategy='constant', fill_value=0)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

    X_test = ctrp_alt_df.loc[module, ctrp_common_columns].T
    X_test = imp_zero.fit_transform(X_test)
    y_test = ctrp_auc.loc[ctrp_common_columns]
    y_test = imp_mean.fit_transform(np.reshape(y_test.values, (-1,1))).ravel()

    test_score = model.score(X_test, y_test)

    return test_score


# Nested CV in GDSC
def nested_gdsc_cv(target_df, alt_df, module,  method, score_param, i_cv=3, o_cv=3, r_cv=2, r_state=0):
    """
    """

    # drug response (y)
    target_df = target_df.dropna(axis=1)
    common_columns = target_df.columns.intersection(alt_df.columns)
    y = target_df.loc[:, common_columns].T
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    y = imp_mean.fit_transform(y).ravel()

    # gene alteration (X)
    X = alt_df.loc[module, common_columns].T
    imp_zero = SimpleImputer(strategy='constant', fill_value=0)
    X = imp_zero.fit_transform(X)

    inner_cv = KFold(n_splits=i_cv, shuffle=True, random_state=r_state)
    outer_cv = RepeatedKFold(n_splits=o_cv, n_repeats=r_cv, random_state=r_state)

    model = create_model("rfr",  score=score_param, cv = inner_cv, n_jobs=10)        
    nested_score = cross_validate(model, X=X, y=y, scoring=score_param, cv=outer_cv, 
                                  return_train_score= True, return_estimator=True)
    return nested_score


def filter_drugs(ctrp_drug_ids, target_prefix, id_drug_dic, ctrp_auc_df, corr_th=0.25):
    """
    filter drugs that are consistent between ctrp and gdsc

    :param ctrp_drug_ids:
    :param target_prefix:
    :param id_drug_dic:
    :param ctrp_auc_df:
    :param corr_th:
    :return:
    """
    corrs, pvs = [], []
    for drug_id in ctrp_drug_ids:
        drug = id_drug_dic[drug_id]
        target_df = pd.read_csv(target_prefix+str(drug_id)+".txt", sep="\t", index_col=0).T.dropna()
        ctrp_drug = ctrp_drug_name(drug)
        ctrp_auc = pd.Series(ctrp_auc_df.loc[ctrp_drug, :]).dropna()
        common_samples = list(filter(lambda x: (x.split("_")[0] in ctrp_auc.index), target_df.index))
        ctrp_map_dic = dict([(x, x.split("_")[0]) for x in common_samples])
        common_ctrp_samples = list(ctrp_map_dic.values())
        target_df.rename(index=ctrp_map_dic, inplace=True)
        corr, pv = ss.spearmanr(target_df.loc[common_ctrp_samples].values, ctrp_auc.loc[common_ctrp_samples].values)
        corrs.append(corr)
        pvs.append(pv)

    data_corr = pd.DataFrame(index=[id_drug_dic[did] for did in ctrp_drug_ids])
    data_corr["drug_id"] = ctrp_drug_ids.values
    data_corr["corr"] = corrs
    data_corr["pv"] = pvs
    corr_drug_ids = data_corr[data_corr["corr"] > corr_th]["drug_id"]
    return data_corr,corr_drug_ids
