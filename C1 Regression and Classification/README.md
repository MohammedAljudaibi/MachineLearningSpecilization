
## Module 1

### Linear regression
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
### Cost Function
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
### Gradient Descent
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

## Module 2
### Multiple Linear Regression
In most cases you're going to need more than one input variable to make an accurate prediction model, in that case multiple w's are going to be needed, one for each input variable x(i). These should be stored in numpy ndarray as computation with them is much more efficient than using python for loops.

![image](https://user-images.githubusercontent.com/121340570/228391451-cd476c63-df4f-41e3-9276-630c7e3e9220.png)

### Feature Scaling
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

### Graded Lab 
This module ended with a [graded lab](C1_W2_Linear_Regression.ipynb) where I made the compute_cost function and the compute_gradient function for a restaurant to predict its profit for a particular city given the city's population size

## Module 3
