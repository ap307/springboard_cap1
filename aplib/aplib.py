import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import linear_model, datasets
from sklearn.decomposition import PCA
import seaborn as sns
#import statsmodels.api as sm
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def fit_plot_distribution(losses, dist_names):
    """
    Function takes a vector and fits a list of statsmodels distributions

    Inputs:
        losses: vector of losses
        dist_names: list of statsodels distributions

    Outputs:
        fig: figure showing histogram of losses and fitted distributions
        dist_fit_dict: dict of statistics for each fitted distribution:
            param: parameters of distribution
            likeL: likeliood value of fitting distribution to data
            AIC: AIC criter of fitting distribution to data
    """
    # Create dictionary to store variables of interest
    dist_fit_dict = {}

    # Set bins
    train_max_loss  = losses.max()
    train_min_loss  = losses.min()
    set_bins = np.logspace(np.log10(train_min_loss), np.log10(train_max_loss), 100)

    # Create figure to store distribution outputs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    y_bins, x_bins, _ = ax.hist(losses,
                            bins=set_bins,
                            facecolor='white',
                            ec='black')

    # Iterate through distributional forms, fit, add to graph and capture Likelihood and AIC information
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(losses)
        likeL = np.sum(getattr(scipy.stats, dist_name).logpdf(losses,
                                                              *param[:-2],
                                                              loc=param[-2],
                                                              scale=param[-1]) )
        cdf_fitted = dist.cdf(set_bins, *param[:-2], loc=param[-2], scale=param[-1]) * losses.size
        pdf_fitted = np.diff(cdf_fitted)
        ax.plot(np.mean([set_bins[1:],set_bins[:-1]], axis=0), pdf_fitted, label=dist_name)
        ax.legend(loc='upper right')
        dist_fit_dict[dist_name] = {"param": param}
        dist_fit_dict[dist_name].update({"likeL": likeL})
        dist_fit_dict[dist_name].update({"AIC": 2*len(param)-2*likeL})

    ax.set_xbound(lower = 0, upper = train_max_loss)
    ax.set_ybound(lower = 0, upper = y_bins.max()*1.1)
    return (fig, dist_fit_dict)

def fit_OLS(losses, ridge_alpha, **kwargs):

    if ('cat' in kwargs) and ('cont' in kwargs):
        dummy_var = pd.get_dummies(kwargs['cat'], drop_first=True)
        all_var = pd.concat([dummy_var, kwargs['cont']], axis=1)
    elif 'cat' in kwargs:
        all_var = pd.get_dummies(kwargs['cat'], drop_first=True)
    else:
        all_var = kwargs['cont']
        
    if ('apply_pca' in kwargs):
        if (kwargs['apply_pca'] == True):
            pca = PCA(whiten=True)
            all_var = pca.fit_transform(all_var)
            # Normalize for comptability with setting ridge regularizaiton factor
            # all_var = normalize(all_var)
            if ('pca_factors' in kwargs):
                num_factors_implement = min(kwargs['pca_factors'], all_var.shape[1])
                all_var = all_var[:,:num_factors_implement]
    else:
        # Normalize prior to regression
        all_var = normalize(all_var)
    
    # Add constant to matrix
    all_var = np.insert(all_var, 0, 1, axis=1)
    
    ols_model = Ridge(alpha=ridge_alpha)
    
    ols_model.fit(all_var, losses)
    loss_pred = ols_model.predict(all_var)

    R2_score = ols_model.score(all_var, losses)
    mean_sq_score = mean_squared_error(loss_pred, losses)
    mean_abs_score = mean_absolute_error(loss_pred, losses)
    
    print "R2: {0:.2f}".format(R2_score)
    print "MSE: {0:.0f}".format(mean_sq_score)
    print "MAE: {0:.0f}".format(mean_abs_score)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=losses, y=loss_pred)
    ax.set_ybound(upper = loss_pred.max())
    ax.set_xbound(upper = losses.max())
    ax.set_title("Predicted vs. Actual losses")

    return (R2_score, mean_sq_score, mean_abs_score, loss_pred, fig)
