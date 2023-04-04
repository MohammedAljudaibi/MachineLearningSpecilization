 # Table of Contents
 - [Module 1](#module-1)
 - - [Neural Network](#neural-network)
 - -
 - [Module 2](#module-2)
 - - []()
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
A simple example of building a Neural Network using TensorFlow can be found in [this lab](C2_W1_Lab02_CoffeeRoasting_TF.ipynb)

The same example of building a Neural Network but without TensorFlow and instead with Numpy can be found in [this optional lab](C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb)

## Graded Lab 1
This Module ended with a [graded lab](C2_W1_Assignment.ipynb) where I practiced how to make a sequential NN model using TensorFlow to predict whether a picture has a handwritten 0 or 1. I also made a similar prediction model but without using TensorFlow. I finally improved the previous function by utilizing matrix multiplication.
