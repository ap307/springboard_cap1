# Capstone Project #1 Submission Overview

**Project Overview**: Submission consists of:

1. Exploratory Data Analysis work book (cap1_eda_review.ipynb): Notebook performs exploratory data analysis on [training set](https://www.kaggle.com/c/allstate-claims-severity/data).

2. XGBoost-based predictive modeling (cap1_predict_review.py): Python script builds predictive model based on XGBoost.

3. Library for storing custom functions (aplib): Contains function to facilitate fitting of alternative distributions to data sets, returns fit criteria and histogram visualization of fit

**Exploratory Data Analysis**

Exploratory data analysis consists of following elements:

1. Fit alternative distributions (list of functions passed to aplib.fit_plot_distribution) and print out fit statistics (AIC and Log Likelihood). Function combines the functionality of several sklearn or stasmodel functions. Fitting is performed prior to and after any transformation of the loss data. An exponential normal function fits the raw data well, therefore a log transform is applied to the raw data (with a shift). Fitting of the transformed losses does not indicate that simple further transformations would improve the normality of the data.

2. A simple screen is run to transform categorical data into numerical data. Two alternative approaches are tested - One Hot encoding and Label Encoding. Simple linear regression is used to identify whether One Hot encoding provides significant advantages over Label Encoding. If not, Label Encoding is used to reduce the size of the feature set to be passed into predictive downstream models. This code was not optimized, and more analysis can be done in terms of determining selection criteria.

3. The distribution of all continuous variables is plotted. There is no clear relationship between any of the individual features and the loss data. The second continuous feature appears to fall into discrete categories and may lend itself to transformation into categorical variables.

4. Test for collinearity of continuous variables. A correlation heat map for all continuous variables plus the loss data is plotted. As represented in the scatter plots, no single feature has a high correlation to the loss data. Some of the continuous variables are highly correlated with each other.

5. Pairs of continuous features showing high correlation (higher than >0.5) are identified and stored in a dictionary. Scatter plots for relationships between those variables are then plotted.

6. To baseline the ability to predict losses using a simple model, a ridge regression model with L2 regularizaiton is fit to the data. The \alpha regularizaiton parameter is optimized through grid search of potential parameters and 10-fold cross validation. The scoring for the search is based on mean absolute error, which is the same evaluation metric as the Kaggle challenge. The optimal \alpha parameter is identified and printed.

7. A PCA analysis is performed to decompose the features into principal components. Test is meant to identify whether the feature data can be compressed without losing predictive capability. Predictive power is tested for 1 to 110 PCA factors. Although cross validation would be a better way to perform this test, with optimized regularizaiton parameters, crude analysis shows that predcitve power increases (albeit not significantly) even when adding additional factors beyond 100 factors. The function of scoring error vs. number of PCA factors included is plotted.

8. Losses are regressed against all features, with the loss transformation inverted. Scatter plot of predicted vs. actual losses is printed.

**XGBoost-based predictive modeling**

Script fits XGBboost model to features. Design elements of the fitting process includes:

- Losses are transformed through a log transformation. Box-Cox transformations were also tested but did not yield appreciably different results, lognormal transformation retained for simplicity. Transform and inverse transform functions are defined.

- A custom scoring function is created for use in cross validation. Since losses may be transformed, a custom function is needed to display results on non-transformed basis.

- The optimization criteria ins mean absolute error, but XGBoost does not provide such an optimization criteria. A custom objective function is implemented. Based on a function that approximates a mean absolute error but is differentiable over all ranges.

- The same selection criteria to determine how categorical features are transformed to numerical features as defined in the exploratory data analysis workbook is applied.

- A 5-fold cross validation analysis to optimize key hyperparameters (through random search over defined ranges) is performed. The GPU-based implementation is used to permit the testing of 200 parameter combinations. Cross validation results are printed.

- Although the GridSearch optimized model is pickled, and could be applied to the test data, the GPU-based XGBoost implementation is less accurate than the CPU-based implementation. Therefore the hyperparameters are re-loaded from the stored cross validation results (instead of using the pickled model directly)

**Discussion of resuts**

The models obtained achieve a mean absolute error on the test data of 1126 to 1140, compared to the top result of 1109. A few elements of the predictive model built for this exercise can be improved on in future iterations:

- Transformations can be applied to the numerical features

- The selection criteria for determining how categorical data is transformed can be optimized

- Different approaches to handle categorical data missing in either train or test data should be tested. The current approach is likely suboptimal.

- Different models such as Neural Nets should be tested (and may be more robust to feature manipulation or loss transformation in pre-processing)
