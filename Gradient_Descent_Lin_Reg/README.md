# Gradient Descent

#### Concepts
Gradient descent is an optimization algorithm used to minimize the cost or loss function of a machine learning model. The goal of gradient descent is to find the values of the parameters that minimize the cost function.

Here's how it works:

1. We start by initializing the parameters of the model to some random values.
2. We then calculate the gradient of the cost function with respect to the parameters. The gradient tells us the direction in which the cost function is increasing the fastest, so we want to move in the opposite direction to minimize the cost function.
3. We update the parameters by subtracting a small amount (known as the learning rate) times the gradient from the current values of the parameters. This is done iteratively until the cost function reaches a minimum.

The learning rate is an important hyperparameter in gradient descent because it determines how big of a step we take in each iteration. If the learning rate is too large, we may overshoot the minimum and the cost function may oscillate or even diverge. If the learning rate is too small, it may take a long time to converge to the minimum.

There are different variations of gradient descent such as batch gradient descent, stochastic gradient descent, and mini-batch gradient descent, each with their own advantages and disadvantages depending on the size and complexity of the dataset.

#### files
| [gradient_descent.py](https://github.com/Keshtech2002/Regression_ML_projects/blob/main/Gradient_Descent_Lin_Reg/gradient_descent.py) | |
| [salary_data.csv](https://github.com/Keshtech2002/Regression_ML_projects/blob/main/Gradient_Descent_Lin_Reg/salary_data.csv) | Practice dataset to check the working principle of my ```gradient descent Linear regression class``` |
| [scratch_linear_reg.ipynb](https://github.com/Keshtech2002/Regression_ML_projects/blob/main/Gradient_Descent_Lin_Reg/scratch_linear_reg.ipynb) | Gradient descent notebook to predict salary and check how model performs |
| [scratch_linear_reg.py](https://github.com/Keshtech2002/Regression_ML_projects/blob/main/Gradient_Descent_Lin_Reg/scratch_linear_reg.py) | Gradient descent class containing ```fit()```, ```predict()```, ```update_weights()``` |