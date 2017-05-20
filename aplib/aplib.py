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
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

def fit_plot_distribution(losses, dist_names, log_scale):
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
    
    # Create figure to store distribution outputs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if log_scale==True:
        set_bins = np.logspace(np.log10(train_min_loss), np.log10(train_max_loss), 100)
        ax.set_xscale("log")
    else:
        set_bins = np.linspace(train_min_loss, train_max_loss, 100)

    # Creaete bins
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
    return fig, dist_fit_dict

def fit_OLS(losses, test_losses, X_train, X_test, ridge_alpha, apply_pca, **kwargs):
    
    all_var = np.array(X_train)
    test_all_var = np.array(X_test)
        
    if apply_pca==True:
        pca = PCA(whiten=False)
        # Apply PCA analysis
        all_var = pca.fit_transform(all_var)
        test_all_var = pca.transform(test_all_var)
        if 'pca_factors' in kwargs:
            num_factors_implement = min(kwargs['pca_factors'], all_var.shape[1])
            all_var = all_var[:,:num_factors_implement]
            test_all_var = all_var[:,:num_factors_implement]
      
    # Add constant to matrix
    all_var = np.insert(all_var, 0, values=1, axis=1)
    test_all_var = np.insert(test_all_var, 0, values=1, axis=1)

    # Fit Ridge regession model to data
    # ols_model = Ridge(alpha=ridge_alpha)
    # ols_model.fit(all_var, losses)
    
    ols_model = GradientBoostingRegressor(loss='lad')
    ols_model.fit(all_var, losses)
    
    loss_pred = ols_model.predict(all_var)
    test_loss_pred = ols_model.predict(test_all_var)

    train_R2 = ols_model.score(all_var, losses)
    train_MABS = mean_absolute_error(loss_pred, losses)
    test_MABS = mean_absolute_error(test_loss_pred, test_losses)
    
    print "Train R2: {0:.2f}".format(train_R2)
    print "Train MSE: {0:.3f}".format(train_MABS)
    print "Test MAE: {0:.3f}".format(test_MABS)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=test_losses, y=test_loss_pred)
    ax.set_ybound(upper = test_loss_pred.max())
    ax.set_xbound(upper = test_losses.max())
    ax.set_title("Test Predicted vs. Test Actual losses")

    return train_R2, train_MABS, test_MABS, fig, loss_pred, test_loss_pred