# Capstone Project #1 Proposal

**Project Overview**: Proposal is to address the [Allstate Claims Severity Kaggle challege](https://www.kaggle.com/c/allstate-claims-severity). This challenge is focused on developing a model to predict claims severity - the amount that is expected to be paid by an insurance company to a policyholder given an event, such as a car collision, that gives rise to a claim - for personal auto insurance policies.

This kind of analysis can be generalized to a broad spectrum of insurance product pricing challenges, e.g. predicting claims frequency, predicting claims severity for other lines of business, predicting how cohorts of policyholders may utilize product features. More specifically, these challenges share a number of common features:

- Large numbers of potential categorical and continuous features that could be used to characterize policyholders and policies, many of which may not have predictive power over all policyholder cohorts and others which may be highly correlated to other features

- Large numbers of historical observations that can range in the millions for large established companies

- Policyholder data is ready available in well formed data sets

**Hypothetical client**: Hypothetical clients may include two distinct functions within an insurance company:

- Underwriting departments, which can use predictions of claims severity to better differentiate pricing for different policyholders cohorts and avoid negative selection (e.g. acquiring or retaining underpriced business that other competitors have appropriately priced)

- Claims departments, which can use predictions of claims severity to identify potentially fraudulent activity (e.g. if submitted claims are outsized compared to predicted claims)

**Data**: Based on data provided as part of the [Kaggle Challenge](https://www.kaggle.com/c/allstate-claims-severity/data). Training data consists of approx. 130,000 records with 116 categorical features and 14 continuous features.

**Project approach**: Preliminary data exploration to include:

- Identification of most likely distributional properties of claims severity data

- Identification of most likely distributional properties of continuous features

- Assessment of overlap / correlation between categorical or continuous features (e.g. to identify what features may be parsimoniously excluded)

- Applying GLM to identify features most likely to have predictive power as a precursor to applying more powerful tools

**Deliverables**: Deliverables to include:

- Description of claims data and features as per exploratory analysis described above, through statistical analyses and graphical representations of distributions and dependencies

- Description of differences in predictive power vs. model complexity for alternative model specificaitons

- Recommendation for single or ensemble of models to be used to predict claims severity

- An assessment of the sensitivity of model selection to different optimization objectives (e.g. mean square error vs. mean absolute error) 