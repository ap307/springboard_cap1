
import os.path
import ast
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy.stats import lognorm as sp_lognorm
from sklearn import linear_model, datasets, preprocessing
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.scorer import make_scorer
import warnings
from sklearn.externals import joblib

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import xgboost as xgb

# Define function to transform (and invesrse transform) losses
log_shift = 700

def transform_losses(losses):
    return np.log(losses + log_shift)

def inv_transform_losses(losses):
    if log_shift_losses==True:
        transform_target = np.float64(np.exp(losses)-log_shift)
        transform_target = np.nan_to_num(transform_target)
        return transform_target
    else:
        return losses

# Create scoring function using inverse transform
def evalerror(y_pred, y_actual):
    return mean_absolute_error(inv_transform_losses(y_pred), inv_transform_losses(y_actual))

evalerror_scorer = make_scorer(evalerror, greater_is_better=False)

# Implement "Fair" objective function (proxy for MAE)
fair_constant = 0.7
def fair_obj(y_actual, y_pred):
    x = (y_pred - y_actual)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess 

# Import data set
cv_file_loc = os.path.expanduser("cv_results.csv")
train_file_loc = os.path.expanduser("data/train.csv")
test_file_loc = os.path.expanduser("data/test.csv")
all_train_data_raw = pd.read_csv(train_file_loc)
all_test_data_raw = pd.read_csv(test_file_loc)

# Load all test data
load_train_rows = all_train_data_raw.shape[0]
all_train_data = all_train_data_raw.loc[:load_train_rows-1,:]
all_train_losses = all_train_data_raw.loc[:load_train_rows-1,"loss"]

# Load all test data
load_test_rows = all_test_data_raw.shape[0]
all_test_data = all_test_data_raw.loc[:load_test_rows-1,:]

# Transform loss data
log_shift_losses=True
if log_shift_losses==True:
    all_train_losses = transform_losses(all_train_losses)
    print "Losses log-transformed"

# Create separate DataFrames for continuous and categorical variables
cat_var = all_train_data.iloc[:,1:117]
cont_var = all_train_data.iloc[:,117:-1]
cat_var_test = all_test_data.iloc[:,1:117]
cont_var_test = all_test_data.iloc[:,117:]

# Simple selection algorithm for deciding whether to use one hot (OH) enconding or label enncoding (LE)
# If predictive power (based on linear regression) of LE is witin 10% of OH encoding, using LE
ratio_threshold = 1.1
le = preprocessing.LabelEncoder()
regr_onehot = linear_model.LinearRegression()
regr_le = linear_model.LinearRegression()
onehot_list = []
label_enc_list = []

for row in cat_var:
    if len(set(cat_var[row]))>2:
        onehot_vec = pd.get_dummies(cat_var[row], drop_first=True)
        label_enc_vec = le.fit_transform(cat_var[row])[:,None]
        regr_onehot.fit(onehot_vec, all_train_losses)
        regr_le.fit(label_enc_vec.reshape(-1,1), all_train_losses)
        ratio = regr_onehot.score(onehot_vec, all_train_losses) / regr_le.score(label_enc_vec.reshape(-1,1), all_train_losses)
        print "Variable: {0}, Number: {1}, One hot / LE score ratio: {2}".format(row, len(set(cat_var[row])), ratio)
        if ratio > ratio_threshold:
            onehot_list.append(row)
        else:
            label_enc_list.append(row)
    else:
        onehot_list.append(row)
        
# Concatenate train and test categorical data for consistent encoding
cat_var_all = pd.concat([cat_var, cat_var_test], axis=0)
labl_enc_dummy_var_all = cat_var_all[label_enc_list].apply(lambda x: le.fit_transform(x))
onehot_dummy_var_all = pd.get_dummies(cat_var_all[onehot_list], drop_first=True)

labl_enc_dummy_var = labl_enc_dummy_var_all.iloc[0:load_train_rows, :]
onehot_dummy_var = onehot_dummy_var_all.iloc[0:load_train_rows, :]
all_train_data = pd.concat([onehot_dummy_var, labl_enc_dummy_var, cont_var], axis=1)

labl_enc_dummy_var = labl_enc_dummy_var_all.iloc[load_train_rows:, :]
onehot_dummy_var = onehot_dummy_var_all.iloc[load_train_rows:, :]
all_test_data = pd.concat([onehot_dummy_var, labl_enc_dummy_var, cont_var_test], axis=1)


# PCA transformation of features (not used)
PCA_transform=False
if PCA_transform==True:
    pca_func = PCA(whiten=False)
    # Apply PCA analysis
    all_train_data = pca_func.fit_transform(all_train_data)
    all_test_data = pca_func.transform(all_test_data)
    print "PCA applied"

# Run random search over XGB parameters
# In future test wider band for min_child_weight
run_searchcv=True
if run_searchcv==True: 
    seed = [1981]
    max_depth_search = [10, 12]
    n_estimators_search = [1000]
    learning_rate_search = [0.01, 0.03]
    min_child_weight_search = [1, 3]
    subsample_search = [0.75, 1]
    colsample_bytree_search = [0.75, 1]
    colsample_bylevel_search = [0.75, 1]
    gamma_search = [1]
    alpha_search = [1]
    XGB_class = xgb.XGBRegressor(objective=fair_obj,
                          n_jobs=1,
                          silent=True,
                          updater='grow_gpu_hist')
    
    rs_search_grid = {'max_depth': sp_randint(7, 12),
    'n_estimators': sp_randint(600, 1300),
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'min_child_weight': [50, 75, 100, 125],
    'subsample': sp_uniform(0.5, 0.5),
    'colsample_bytree': sp_uniform(0.7, 0.3),
    'colsample_bylevel':  sp_uniform(0.7, 0.3),
    'gamma': [1, 2, 2.5],
    'reg_alpha': [1.5, 2, 2.5, 3],
    'seed': seed
    }
    cv = RandomizedSearchCV(XGB_class, verbose=500, scoring=evalerror_scorer, cv=3,
                            n_iter=200, param_distributions=rs_search_grid, random_state=1, refit=True)
    now = datetime.now()
    cv.fit(all_train_data, all_train_losses)
    pd.DataFrame(data=cv.cv_results_).to_csv(cv_file_loc)
    print cv.cv_results_
    print "Saving best estimator"
    joblib.dump(cv.best_estimator_, 'RandomSearchXGB.pkl')

print "Loading best estimator"
xgb_estimator = joblib.load('RandomSearchXGB.pkl')

print "Loading best estimator parameters"
cv_results = pd.read_csv(cv_file_loc)
xgb_best_params = ast.literal_eval(cv_results[cv_results["rank_test_score"]==1]["params"].iloc[0])

print "Re-fitting XGB with CPU only"
XGB_class_refit = xgb.XGBRegressor(objective=fair_obj,
                          n_jobs=-1,
                          silent=False,
                          **xgb_best_params)
XGB_class_refit.fit(all_train_data, all_train_losses)
all_test_loses_pred = inv_transform_losses(XGB_class_refit.predict(all_test_data))

# Save results
now = datetime.now()
ids = all_test_data_raw.iloc[:,0].values.astype(np.int32)
result_fixed = pd.DataFrame(all_test_loses_pred, columns=['loss'])
result_fixed["id"] = ids
result_fixed = result_fixed.set_index("id")
sub_file = 'submission_RS-CV-xgb_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing submission: %s" % sub_file)
result_fixed.to_csv(sub_file, index=True, index_label='id')


