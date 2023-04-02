
 # Table of Contents
 - [Module 1](#module_1)
 - - [Linear Regression](#linear-regression)
 - - [Cost Function](#cost-function)
 - - [Gradient Descent](#gradient-descent)
 - [Module 2](#module-2)
 - - [Multiple Linear Regression](#multiple-linear-regression)
 - - [Feature Scaling](#feature-scaling)
 - - [Graded Lab 1](#fraded-lab-1)
 - [Module 3](#module_3)
 - - [Logistic Regression](#logistic-regression)
 - - [Cost Function for Logistic Regression](#cost-function-for-logistic-regression)
 - - [Gradient Descent for Logistic Regression](gradient-descent-for-logistic-regression)
 - - [Overfitting](#overfitting)
 - - - [Addressing Overfitting](#addressing-overfitting)
 - - [Regularization](#regularization)


# Module 1

## Linear Regression
Linear regression is a supervised machine learing model that is used to predict the value of variable based on the values of other variables

![image](https://user-images.githubusercontent.com/121340570/228200933-1d267f88-ca3a-4b48-b9ea-7482c1b1f587.png)

This formula for linear regression gives the prediction for the i-th data point using parameters w,b. 
In python code:
```python
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb
``` 
## Cost Function
Cost or J(w,b)  aims to minimize the error between the predictions over all the training samples.
To get w and b values a cost function such as this can be used:

![image](https://user-images.githubusercontent.com/121340570/228202576-be32d43f-da1e-46de-b73a-b07fef015c18.png)

In python code: 
```python
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost
```
## Gradient Descent
Gradient descent aims to minimize the cost function by converging from the initial w,b values until it reaches their optimal values.
Gradient is defined as:

![image](https://user-images.githubusercontent.com/121340570/228208112-2769fd2d-fb4d-4341-b4e5-891b686217c3.png)

Python code for computing gradient:
```python
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db
```
[This Lab](C1_W1_Lab04_Gradient_Descent_Soln.ipynb) showcases all the previous functions and formulas and uses them in a gradient descent learning algorithm to predict the cost of a hose given its size in square feet.

To review and practice what I learnt in this module I went back to an [old project](https://github.com/MohammedAljudaibi/TMDB-Analysis-DAND2) and I made a [simple linear prediction model](https://github.com/MohammedAljudaibi/MachineLearningS/blob/main/C1%20Regression%20and%20Classification/TMDb%20Gradient%20Descent%20.ipynb) to predict how much revenue a movie would make given its average movie rating. 

![image](https://user-images.githubusercontent.com/121340570/228288723-954589b1-e62a-4d63-b5b2-fdbb8817d988.png)

# Module 2
## Multiple Linear Regression
In most cases you're going to need more than one input variable to make an accurate prediction model, in that case multiple w's are going to be needed, one for each input variable x(i). These should be stored in numpy ndarray as computation with them is much more efficient than using python for loops.

![image](https://user-images.githubusercontent.com/121340570/228391451-cd476c63-df4f-41e3-9276-630c7e3e9220.png)

## Feature Scaling
Features that are too large or too small can cause linear regression to take longer to run, therefore its beneficial to rescale to features and have them all around 0 or 1, for example by dividing each feature by the max in said feature. Rescaling or normalization can be done in two more ways, mean normalization and Z-score normalization.
Z-score normalization formulas and code:

![image](https://user-images.githubusercontent.com/121340570/228393806-b3618027-3486-4b4a-8424-80a9f934e8ae.png)

```python
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example
    #  then divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
```

## Graded Lab 1
This module ended with a [graded lab](C1_W2_Linear_Regression.ipynb) where I made the compute_cost function and the compute_gradient function for a restaurant to predict its profit for a particular city given the city's population size

# Module 3
## Logistic Regression
Linear regression is used for numeric models, where the output is a range of numbers. Logistic regression can be inaccurate for models which have a classification problem, a binary output that is 0 or 1, false or true, or an output that is ordinal such as a students grades (A, B, C, D, F). For such problems logistic regression should be used, it is better suited to make classification models.
Formula and sigmoid function g(z):

![image](https://user-images.githubusercontent.com/121340570/228693395-3b52c4a1-7517-4754-b147-cce4d3ed2809.png)

The formula is similar to the linear regression formula, but it is inputted into the sigmoid function.
Implementation of the Sigmoid function:
```python
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """

    g = 1/(1+np.exp(-z))
   
    return g
```

The output of this function will be between 0 and 1, this can be looked at like a percentage, if the output is 0.89 then it can be interpreted as the model predicts with 89% certainty that X is true. a heuristic should be used to predict at which threshold should y=0 be predicted, something such as this:

![image](https://user-images.githubusercontent.com/121340570/228694376-12c0a4a2-e66f-47fd-9918-b24869a475ba.png)

## Cost Function for Logistic Regression
The cost function for linear regression cannot be used for logistic regression as it can be erratic and give many local minimums, instead logistic regression has its own loss function which is:

![image](https://user-images.githubusercontent.com/121340570/228879332-d88b48ab-c0de-4a01-b950-242cc8bc5f24.png)

and its implementation in python:  

```python
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost
```
## Gradient Descent for Logistic Regression
Logistic regression has the same two formulas for gradient descent as shown [previously](#GradientDescent), the only difference is that fw,b now has the sigmoid function. 
Implementation in python:
```python
def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
        
    return dj_db, dj_dw  
```
## Overfitting

Overfitting happens when the algorithm fits the data *too* well, almost as if it remembers exactly how each of training data points are and can only accurately predict for them, and is bad at predicting new data

![Overfitting: Causes and Remedies – Towards AI](https://cdn-images-1.medium.com/max/395/1*o9s_R6V2Z0piFzoABu0W2Q.png)

### Addressing Overfitting
The following can help minimize overfitting
1. Collecting more data.
2. Feature selection, pick the most useful parameters and disregard the rest.
3. Regularization, instead of disregarding parameters that aren't as useful, give them less weight.

## Regularization

As mentioned previously, regularization helps to prevent overfitting. Regularization can be added to the cost function and both linear and logistic methods. For example regularization can be added to the loss function by adding the following to it:

![image](https://user-images.githubusercontent.com/121340570/228920340-66b89700-cfff-48eb-a2ce-faa6b50f3d89.png)

Then this code can be added into the cost function where the total_cost would be cost + reg_cost
```python
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost
```


## Graded Lab 2
This module and course ended with a [graded lab](C1_W3_Logistic_Regression.ipynb) where I made the sigmoid and compute cost functions for logistic regression. I also implemented the compute_gradient for logistic regression, I then predicted whether the label is 0 or 1. I also made a function to compute the cost using regularization, I then finally made combined everything to compute the gradient using regularization.
