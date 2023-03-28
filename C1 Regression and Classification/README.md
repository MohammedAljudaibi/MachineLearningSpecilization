
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


## Module 2

## Module 3
