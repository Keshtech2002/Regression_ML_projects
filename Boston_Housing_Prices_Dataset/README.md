# Boston_Housing_Prices_Dataset

### ABOUT THE PROJECT
Taking some inferences from the boston housing data and building a multiple linear regression model.
#### Concepts mentioned here were learnt and utilized in this project

#### Exploratory Data Analysis (EDA)

#### Regression
- Regression analysis
- Linear Regression
- Simple Linear Regression
- Multiple Linear Regression
- Residuals: Difference between predicted and actual values of the target variables
- Coefficients: Which predictor variables have most impact on the target variable
- Ordinary Least squares: OLS to estimate the regression coefficients and minimize the sum of squared residuals
- R-squared: Higher R-squared value indicates a better fit, determine how model fits the data
- Overfitting: Avoid overfitting with techniques such as cross-validation and regularization
- Regularization: L1, L2 regularization techniques to prevent overfitting and improve the generalizability of the model
- Gradient Descent: Optimize model parameters and improve the fit of the model

#### Other concepts
- What are Observational and Experimental data?
- What are Lurking Variables?
- The Gauss Markov Theorem
- Sampling distributions of Regression coefficients
- Anova Partitioning
- Diagnostic and Remedial Measures
- Types of Regression
- Correlation and Causation
- Formula for Regression
- Understanding Interpolation and Extrapolation
- Derivation for Least Square Estimates
- Point estimators of Regression
- F- Statistics
- Coefficient of Determination(R-Squared)
- General Linear Regression Model
- Matrix Representation for General Linear Regression Model
- Matrix Representation of Least Squares
- Understanding types of predictive variables
- F-test
- Coefficient of multiple determination
- Adjusted R- squared
- What are scatterplots?
- What is a correlation matrix?
- Understanding multicollinearity.
- Anova Partitioning
- Diagnostic and Remedial Measures
- What are Indicator Variables?
- Various criteria for model selection such as R^2, mallows cp Criterion, AIC/SBC Criterion, Press Criterion

### Files
| File | |
| -------- | ---- |
| [Housing.ipynb](https://github.com/Keshtech2002/Regression_ML_projects/blob/main/Boston_Housing_Prices_Dataset/Housing.ipynb) | Exploratory analysis and Regression notebook |
| [HousingData.csv](https://github.com/Keshtech2002/Regression_ML_projects/blob/main/Boston_Housing_Prices_Dataset/HousingData.csv) | Boston Housing Dataset |
| [Untitled.ipynb](https://github.com/Keshtech2002/Regression_ML_projects/blob/main/Boston_Housing_Prices_Dataset/Untitled.ipynb) | Rough note |

### ABOUT THE DATA
- **Number of Instances**: 506
- **Number of Features**: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

- **Features Info**
    - CRIM - per capita crime rate by town
    - ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    - INDUS - proportion of non-retail business acres per town.
    - CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    - NOX - nitric oxides concentration (parts per 10 million)
    - RM - average number of rooms per dwelling
    - AGE - proportion of owner-occupied units built prior to 1940
    - DIS - weighted distances to five Boston employment centres
    - RAD - index of accessibility to radial highways
    - TAX - full-value property-tax rate
    - PTRATIO - pupil-teacher ratio by town
    - B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    - LSTAT - % lower status of the population
    - MEDV - Median value of owner-occupied homes in $1000's (Target Variable)