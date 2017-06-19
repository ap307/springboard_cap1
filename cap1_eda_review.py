
# coding: utf-8

# In[1]:

import os.path
import sys
from inspect import getsourcefile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import linear_model, datasets
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import xgboost as xgb

# Custom modules
from aplib import aplib

# In[2]:

# Customizable function to transform loss data
def transform_losses(losses):
    return np.log(losses + 200)


# In[3]:

# Import data set
train_file_loc = os.path.expanduser("data/train.csv")
all_data = pd.read_csv(train_file_loc)

# Test data provided does not have losses, so use CV to create test sample
#test_file_loc = os.path.expanduser("~/Documents/springboard_files/springboard_cap1/data/test.csv")
#test_data = pd.read_csv(test_file_loc)

# Print sample data
all_data.head(5)


# In[4]:

# Load all data
#load_rows = 20000
load_rows = all_data.shape[0]
train_data = all_data.loc[:load_rows-1,:]
train_losses = all_data.loc[:load_rows-1,"loss"]
all_train_losses = all_data.loc[:load_rows-1,"loss"]

# Create separate DataFrames for continuous and categorical variables
cat_var = train_data.iloc[:,1:117]
cont_var = train_data.iloc[:,117:-1]

# Encode categorical variables, re-join to continuous variables
dummy_var = pd.get_dummies(cat_var, drop_first=True)
train_data = pd.concat([dummy_var, cont_var], axis=1)
all_train_data = pd.concat([dummy_var, cont_var], axis=1)


# In[5]:

# Load test data
#test_load_rows = test_data.shape[0]
#test_data = test_data.loc[:test_load_rows-1,:]
#test_losses = test_data.loc[:test_load_rows-1,"loss"]
train_data, test_data, train_losses, test_losses = train_test_split(train_data,
                                                                    train_losses,
                                                                    test_size=0.1,
                                                                    random_state=3)
test_load_rows = test_data.shape[0]


# In[6]:

box_cox_losses=True

if box_cox_losses==True:
    train_losses, box_cox_lambda = scipy.stats.boxcox(train_losses, lmbda=None)
    test_losses = scipy.stats.boxcox(test_losses, lmbda=box_cox_lambda)
    all_train_losses = scipy.stats.boxcox(all_train_losses, lmbda=box_cox_lambda)
    print box_cox_lambda
    
def invboxcox(y,ld):
    return (ld*y+1)**(1/ld)


# In[7]:cd .. 

max_depth_search = [2]#3, 5, 10]
n_estimators_search = [5]
learning_rate_search = [0.01]#0.01, 0.05, 0.1]
min_child_weight_search = [1, 3]#3, 5, 10]


# In[ ]:

def evalerror(y_pred, dmatrix):
    return 'mae_adj', mean_absolute_error(invboxcox(y_pred, box_cox_lambda),
                                          invboxcox(dmatrix.get_label(), box_cox_lambda))
    
RANDOM_STATE = 2016

params = {
    'objective': 'reg:linear',
    'min_child_weight': 1,
    'eta': 0.01,
    'colsample_bytree': 0.5,
    'max_depth': 12,
    'subsample': 0.8,
    'alpha': 1,
    'gamma': 1,
    'silent': 1,
    'verbose_eval': True,
    'seed': RANDOM_STATE
}

kf = KFold(n_splits=3)
kf.get_n_splits(train_losses)

print kf

xgtrain = xgb.DMatrix(train_data, label=train_losses)
xgtest = xgb.DMatrix(test_data, label=test_losses)

model = xgb.cv(params, xgtrain, 2000, show_stdv=True, seed=1, stratified=True, verbose_eval=True,
               folds=kf, as_pandas=True, feval=evalerror, nfold=3) #evals=[(xgtest,'eval')]

model.to_csv("xgbcv_results.csv")

#--------- Dormant GS CV -------------------------------------------------------
## In[ ]:
#
#cv = GridSearchCV(XGB_class, {'max_depth': max_depth_search,
#                                   'n_estimators': n_estimators_search,
#                                   'learning_rate': learning_rate_search,
#                                  'min_child_weight': min_child_weight_search},
#    verbose=10)
##                'early_stopping_rounds':20,
##                'eval_set':(test_data, test_losses)})
#
#args = {'eval_metric':'error'}
#cv.fit(train_data, train_losses, **args)
#
#pd.DataFrame(data=cv.cv_results_).to_csv("cv_results.csv")
#
#print cv.cv_results_

## In[7]:
#
#--------- Dormant Scikit -------------------------------------------------------
#print "Start XGBoost fit"
#
#
#XGB_class = xgb.XGBRegressor(max_depth=2,
#                              objective='reg:linear',
#                              n_estimators=2,
#                              min_child_weight=5,
#                              n_jobs=10,
#                              silent=True,
#                              learning_rate=0.01)
#
#clf = XGB_class.fit(train_data,
#                    train_losses,
#                    eval_metric=evalerror,
#                    eval_set=[(test_data, test_losses)])
#
##your turn. Print the accuracy on the test and training dataset

#training_accuracy = mean_absolute_error(invboxcox(train_losses, box_cox_lambda),
#                                        invboxcox( model.predict(xgtrain), box_cox_lambda))
#
#test_accuracy = mean_absolute_error(invboxcox(test_losses, box_cox_lambda),
#                                        invboxcox( model.predict(xgtest), box_cox_lambda))
#
#
#print("Accuracy on training data: {:2f}".format(training_accuracy))
#print("Accuracy on test data:     {:2f}".format(test_accuracy))

#
#
## In[5]:
#
## Split categorical and continuos variables
#cat_var_list = [col for col in list(train_data) if col.startswith('cat')]
#cont_var_list = [col for col in list(train_data) if col.startswith('cont')]
#
#
## In[6]:
#
#
#
#
## In[ ]:
#
## Check for completenes of data, returns 0 if there are entries for each loss data point
#if sum(train_data.count(axis=0) != train_losses.size):
#    print "Missing data mapped to losses"
#else:
#    print "No missing data mapped to losses"
#
#
## In[ ]:
#
## For categorical variables, create dictionary of unique values
#unique_dict = {}
#for column in cat_var:
#    unique_dict[column]=cat_var[column].unique()
#
#
## In[ ]:
#
## List of distributional forms to explore
#dist_names = ['norm']#,'lognorm','exponnorm', 'gamma']#, 'beta','norm', 'beta', 'rayleigh', 'norm', 'pareto']
#
## Call function to plot distribution and return fit information
#figure, dist_info = aplib.fit_plot_distribution(train_losses, dist_names, log_scale=False)
#print dist_info
#
#
## In[ ]:
#
## Plot losses vs. continuous variables
#fig, axes = plt.subplots(7,2, figsize=(12,24))
#
#for counter, col in enumerate(cont_var_list):
#    axes[counter // 2][counter % 2].scatter(x=train_data[col], y=train_losses)
#    axes[counter // 2][counter % 2].set_title(col)
#
#fig.tight_layout()
#
#
## In[ ]:
#
## Plot correlation matrix heat map
#cont_var_wloss = all_data.iloc[:load_rows-1,117:]
#corr = cont_var_wloss.corr()
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#f, ax = plt.subplots(figsize=(11, 9))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
#ax = sns.heatmap(corr, mask=mask, cmap=cmap,
#            square=True, xticklabels=True, yticklabels=True,
#            linewidths=.25, cbar_kws={"shrink": .75}, ax=ax)
#
#ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
#ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
#
#
## In[ ]:
#
## Identify variables with a degree of correlatin > threshold (between themselves or vs. loss)
#
#threshold = 0.5
#sig_dict = {}
#
#sig_counter = 0
#for counter1 in range(0, cont_var_wloss.columns.size):
#    for counter2 in range(counter1 + 1, cont_var_wloss.columns.size):
#        correl, pval = scipy.stats.pearsonr(cont_var_wloss.iloc[:,counter1], cont_var_wloss.iloc[:,counter2])
#        if (correl<=-threshold) or (correl>=threshold):
#            sig_dict[sig_counter] = {(cont_var_wloss.columns[counter1]):(cont_var_wloss.columns[counter2])}
#            sig_counter += 1
#
#sig_dict
#
#
## In[ ]:
#
## Plot subset of continous variables and loss with a degree of correlation > threshold
#fig, axes = plt.subplots(int(np.ceil((len(sig_dict)+1)/2)),2, figsize=(12,48))
#
#for key, value in sig_dict.iteritems():
#    title = ('{} vs {}').format(value.items()[0][0],  value.items()[0][1])
#    axes[key // 2, key % 2].scatter(train_data[value.items()[0][0]], train_data[value.items()[0][1]])
#    axes[key // 2, key % 2].set_title(title)
#
#fig.tight_layout()
#
#
## In[ ]:
#
## Define cross validation scheme
#kf_10 = KFold(n_splits=10, shuffle=False, random_state=2)
#
## Transform variables through PCA
#cv_pca = PCA(whiten=False)
#
## Run through options for alpha parameter
#regr = Ridge(fit_intercept=True)
#param_grid = {'alpha': [10, 1000]}
#ridge_param_eval = GridSearchCV(regr, param_grid, cv=kf_10, scoring='neg_mean_absolute_error')
#ridge_param_eval.fit(cv_pca.fit_transform(np.array(all_train_data)), np.array(all_train_losses))
#ridge_param_eval.cv_results_
#
#
## In[ ]:
#
## Print out optimum parameter
#opt_param = ridge_param_eval.cv_results_['params'][ridge_param_eval.cv_results_['rank_test_score'].argmin()]
#print opt_param
#
#
## In[7]:
#
## Optimal regression parameters based on analysis below
## For analyses based on PCA where variance of vectors is important, num_pca controls max number of features considered
#
#ridge_reg_alpha = 10 #opt_param['alpha']
#num_pca = []
#
#
## In[10]:
#
## Run all variables through OLS
#get_ipython().magic(u'timeit')
#train_R2, train_MABS, test_MABS, test_scatter_plot, loss_pred, test_loss_pred = aplib.fit_OLS(losses=train_losses,
#                                          test_losses=test_losses,
#                                          X_train=train_data,
#                                          X_test=test_data,
#                                          apply_pca=True,
#                                          ridge_alpha=ridge_reg_alpha)
#
#
## In[11]:
#
#def invboxcox(y,ld):
#    return (ld*y+1)**(1/ld)
#
#if box_cox_losses==True:
#    loss_pred = invboxcox(loss_pred, box_cox_data_lambda)
#    test_loss_pred = invboxcox(test_loss_pred, box_cox_data_lambda)
#    test_losses = invboxcox(test_losses, box_cox_data_lambda)
#    train_losses = invboxcox(train_losses, box_cox_data_lambda)
#
#
## In[12]:
#
#from sklearn.metrics import mean_absolute_error
#print mean_absolute_error(test_loss_pred, test_losses)
#print mean_absolute_error(loss_pred, train_losses)
#plt.scatter(x=loss_pred, y=train_losses)
#
#
## In[11]:
#
## For analysis where variance of input data is an important feature does not whiten feature data
#opt_regr = Ridge(alpha=ridge_reg_alpha)
#max_factors = 110
#
#dummy_var = pd.get_dummies(cat_var, drop_first=True)
#all_var = pd.concat([dummy_var, cont_var], axis=1)
#
## Identify PCA
#pca = PCA(whiten=False)
#all_var = pca.fit_transform(all_var)
#all_var.shape
#
## Add constant
#all_var = np.insert(all_var, 0, 1, axis=1)
#
#mse = []
#for i in np.arange(1,max_factors+2):
#    score = -cross_val_score(opt_regr, all_var[:,:i], 
#                             all_data.loc[:load_rows-1,"loss"], 
#                             cv=kf_10, scoring='neg_mean_absolute_error').mean()
#    mse.append(score)
#    print 'Up to {0} PCA factors analyzed\r'.format(i)
#
#
## In[ ]:
#
#import matplotlib.ticker
#fig,ax = plt.subplots()
#ax.plot(np.arange(1,max_factors+2), mse[0:])
#ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
#ax.set_xlabel("Number of PCA factors")
##ax.set_xticks(np.arange(1,max_factors+1))
#ax.set_ylabel("Mean Abasolute Error")
#ax.set_title("Mean Abasolute Error vs. Number of PCA Factors")
#
#
## In[ ]:
#
#
#
#
## In[ ]:



