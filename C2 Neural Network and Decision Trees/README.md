 # Table of Contents
 - [Module 1](#module-1)
 - - [Neural Network](#neural-network)
 - - [TensorFlow Implementation of NNs](#TensorFlow-Implementation-of-NNs)
 - - [Graded Lab 1](#Graded-Lab-1)
 - [Module 2](#module-2)
 - - [Activation Functions](#Activation-Functions)
 - - [Multiclass Classification](#Multiclass-Classification)
 - - [Optimizing Cost Function](#Optimizing-Cost-Function)
 - - [Additional Layer Types](#Additional-Layer-Types)
 - - [Graded Lab 2](#Graded-Lab-2)
 - [Module 3](#module-3)
 - - []()
 - [Module 4](#module-4)
 - - []()

 # Module 1
 
 ## Neural Networks
 A neural network works by having its input features go through some number of hidden layers before finally giving the output/s, each layer has several neurons which can be more, less, or the same number as the input features. 

![neutral-network-diagram](https://user-images.githubusercontent.com/121340570/229573834-9a80b440-a072-4fcf-898c-a0d312904e75.svg)

Each neuron would basically apply a function on it's inputs, for example logistic regression, and then it uses that output as the inputs for the next layer, until it finally uses the last hidden layer as inputs to the output layer and make the prediction.

## TensorFlow Implementation of NNs
How to make a neural network to predict whether a hand written numbere is 0 or 1: 
The input for here would be images instead of numbers, but for computers images are also just numbers, pixels are just values from [0-255], and image that is 16x16 pixels would have 256 pixels or as a computer would read it, 256 values between [0-255].

Now the neural network needs to be designed by seeing how many layers should be used and how many neurons should be in each layer, let's say it should have 2 hidden layers, the first can have 30 neurons and the second can have 20. The last layer would be the output layer and it would have only 1 neuron, the prediction that that is a 0 or 1 in percentage.

```py
layer1 = Dense(units=30, activation='sigmoid')
layer2 = Dense(units=20, activation='sigmoid')
layer3 = Dense(units=1, activation='sigmoid')
```
To connect the layers together to form a NN, TensorFlow's Sequential function and compile function can be used:
```py
model = Sequential([layer1, layer2, layer3])
#layers do not need to be declared before the sequential function
#the model can also be used like this:
model = Sequential([
	Dense(units=30, activation='sigmoid') #hidden layer 1
	Dense(units=20, activation='sigmoid') #hidden layer 2
	Dense(units=01, activation='sigmoid') #output layer or layer 3
	])
	
model.compile( #this function will be explained later
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)
```

The input and outputs are also needed:
```py
x = np.array([0,..., 255,..., 12],
			 [0,..., 124,..., 54])
y = np.array([1,0])
```
Finally, only the model fitting and predicting is left:
```py
model.fit(x,y)
model.predict(x_new) #where x_new are the pictures you want to predict (0 or 1)
```

This shows that a Neural Network in TensorFlow is split into three main
1. Model: Creating the model
2. Compile: Configuring the model and preparing it for training
3. Fit: Training the model

A simple example of building a Neural Network using TensorFlow can be found in [this lab](C2_W1_Lab02_CoffeeRoasting_TF.ipynb)

The same example of building a Neural Network but without TensorFlow and instead with Numpy can be found in [this optional lab](C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb)

## Graded Lab 1
This Module ended with a [graded lab](C2_W1_Assignment.ipynb) where I practiced how to make a sequential NN model using TensorFlow to predict whether a picture has a handwritten 0 or 1. I also made a similar prediction model but without using TensorFlow. I finally improved the previous function by utilizing matrix multiplication.

# Module 2

## Activation Functions
Activation functions are the functions that are used between  and to connect all the neurons. In module 1, only the sigmoid function was used as the activation function in both the hidden and output layers. 

Some common activation functions for the output layer (last layer in NN) and when its best to use them:
- **Sigmoid Function**: used for **binary** classification where the output can be 0 or 1
- **Linear Function**: used in **regression** problems where the output can be any negative or postive number
- **ReLU Function**[ g(z)=max(0,z) ]: also used in **regression** problems but when the output can only be postive

For the hidden layers it is best use the ReLU function, it is both faster to compute and it is better optimized for gradient descent.

The neural network in the Module 1 should now look like this:
```py
model = Sequential([
	Dense(units=30, activation='relu') #hidden layer 1
	Dense(units=20, activation='relu') #hidden layer 2
	Dense(units=1, activation='sigmoid') #output layer or layer 3
	])
``` 

The linear function should never be used for neural networks as it basically defeats the purpose of them and just makes them equivalent to linear or logistic regression.
 
## Multiclass Classification
The Softmax regression algorithm is a generlization of logistic regression, that computes the probability of how each output is likely to be true. To predict more than one class, the final output layer in the NN would need to have more than 1 unit.

The softmax function can be written as:
 ![image](https://user-images.githubusercontent.com/121340570/230383860-1dc4fb41-d709-4e01-9b8d-786296af0b86.png)

The code for making a muticlass NN using softmax would look like this
```py
model = Sequential([
	Dense(units=30, activation='relu') #hidden layer 1
	Dense(units=20, activation='relu') #hidden layer 2
	Dense(units=1, activation='softmax') #output layer or layer 3
	])

model.compile(loss=SparseCategoricalCrossEntropy())
```
<sub><sup>Note: The Sprase Categorical Cross Entropy is used only when one of the many output can true, for example when trying to predict a handwritten number from 0 to 9, the picture can and only has one number</sup></sub>
But this code can lead to computational errors in python when the _z_ is too large or too small, instead the following code does exactly the same as the previous code but helps reduce the computational error.

```py
model = Sequential([
	Dense(units=30, activation='relu') #hidden layer 1
	Dense(units=20, activation='relu') #hidden layer 2
	Dense(units=1, activation='linear') #output layer or layer 3
	])

model.compile(loss=SparseCategoricalCrossEntropy(from_logits=True))
```

By doing this the last layer was changed from the softmax function to the linear function, but the softmax still needs to be used to accurately predict and categorize the output, this can be using the following code:
```py
logits = model(X)
f_x = tf.nn.softmax(logits)
```

[This lab](C2_W2_Multiclass_TF.ipynb) visualizes how multiclass predictions work in TensorFlow

## Optimizing Cost Function
When using gradient descent to find the minimum of the cost function, when the learning rate is too small then it will take longer to converge, and when the learning rate is too big the function will bounce and may overshoot or diverge away from the minimum

![5: Effect of the learning rate on SGD updates. Too small (left) may... |  Download Scientific Diagram](https://www.researchgate.net/publication/323218981/figure/fig6/AS:594583624884225@1518771191628/Effect-of-the-learning-rate-on-SGD-updates-Too-small-left-may-take-longer-to.png)


The Adam Algorithm (**Ada**ptive **M**oment estimation) works to solve both of these issues. It increase the learning rate when it moves in the same direction, and decreases it when it keeps oscillating.

The Adam algorithm can be used in an NN by adding it to the model.compile parameters:
```py
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), ...)
```
<sub><sup>Note: any can learning rate can be used here not only 0.001, it completely depends on the dataset and cost function, but usually, smaller is better</sup></sub>

## Additional Layer Types
Previously in the course, only Dense layers were used to make models, in Dense layers each neuron output is used in all the neurons in the next layer. Another layer type is the **Convolutional** layer which basically has filters to look at images. For example instead of just looking at an image one pixel at a time, instead the convolutional layer can look at image in a 3x3 viewing window. The viewing window can be made to only look at certain pixels for example, only the pixels in the diagonal of the 3x3 viewing window. Convolutional layers are faster to compute compared to Dense layers, and Convolutional Neural Networks can help to combat overfitting when the training data is small. 

![ML Practicum: Image Classification | Machine Learning | Google Developers](https://developers.google.com/static/machine-learning/practica/image-classification/images/convolution_example.svg)

## Graded Lab 2
In [This Graded Lab](C2_W2_Assignment.ipynb) I made a three layer neural network to predict and classify handwritten numbers from 0 to 9.
The NN correctly classified 4985 images out of a total of 5000.

![image](https://user-images.githubusercontent.com/121340570/230404877-8d4c946c-3af7-45cf-a9f9-021365a7c49a.png)

