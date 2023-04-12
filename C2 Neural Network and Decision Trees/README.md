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
 - - [Evaluating a Model](#Evaluating-a-Model)
 - - [Bias and Variance](#Bias-and-Variance)
 - - [Combatting High Bias and Variance](#Combatting-High-Bias-and-Variance)
 - - [Machine Learning Development Process](#Machine-Learning-Development-Process)
 - - - [Iterative Loop](#Iterative-Loop)
 - - - [Error Analysis](#Error-Analysis)
 - - - [Data Augmentation](#Data-Augmentation)
 - - - [Transfer Learning](#Transfer-Learning)
 - - [Error for Skewed Datasets](#Error-for-Skewed-Datasets)
 - - [Graded Lab 3](#Graded-Lab-3)
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
	Dense(units=30, activation='sigmoid'), #hidden layer 1
	Dense(units=20, activation='sigmoid'), #hidden layer 2
	Dense(units=1, activation='sigmoid') #output layer or layer 3
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
	Dense(units=30, activation='relu'), #hidden layer 1
	Dense(units=20, activation='relu'), #hidden layer 2
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
	Dense(units=30, activation='relu'), #hidden layer 1
	Dense(units=20, activation='relu'), #hidden layer 2
	Dense(units=1, activation='softmax') #output layer or layer 3
	])

model.compile(loss=SparseCategoricalCrossEntropy())
```
<sub><sup>Note: The Sprase Categorical Cross Entropy is used only when one of the many output can true, for example when trying to predict a handwritten number from 0 to 9, the picture can and only has one number</sup></sub>
But this code can lead to computational errors in python when the _z_ is too large or too small, instead the following code does exactly the same as the previous code but helps reduce the computational error.

```py
model = Sequential([
	Dense(units=30, activation='relu'), #hidden layer 1
	Dense(units=20, activation='relu'), #hidden layer 2
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

![image](https://user-images.githubusercontent.com/121340570/230408188-1740f5b8-a7df-4c18-98ab-0f18be639ab3.png)

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

# Module 3

## Evaluating a Model

The dataset for a model can be split into training and test sets, and the test test can be further split into cross validation/dev set and test set. The training set should always be the largest set, for example 80% training set, 10% dev test and 10% test set. After training the data, dev set is used to test the cost of the model, and finally test set is used to test the accuracy of the model. 

Test set alone can be used to test the cost and the accuracy but this could lead to generlization and that is why the dev test should be used instead. In [this model evaluation lab](C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb), different models where evaluated, both regression with different orders of polynomiyals, and neural networks with different layers, to find which is has the lowest error and is the better option.

## Bias and Variance

High bias causes **underfitting**, and high variance causes **overfitting**

![Bias and Variance example](https://user-images.githubusercontent.com/121340570/230628739-3eb52700-5813-4d05-9bb0-2eee64b7aac3.PNG)

This simple table shows how the cost or error of the train and dev sets can show if the model is underfitting or overfitting

||underfitting|optimal| overfitting|
|:---:|:---:|:---:|:---:|
|J<sub>train</sub>	|high|low|low
|J<sub>CV</sub>		|high|low|high

While uncommon, it is possible for your model to have both high bias and high variance, this can happen when J<sub>train</sub> is high, and J<sub>CV</sub> is much higher the J<sub>train</sub>. This usually is caused by the model overfit for a portion of the dataset, and then underfit for another portion of the dataset.

## Combatting High Bias and Variance
More training data → Fixes high variance
Less sets of features → Fixes high variance
Getting more features → Fixes high bias
Adding polynomial features (x<sub>1</sub>x<sub>2</sub>, x<sub>1</sub><sup>3</sup>) → Fixes high bias
Decreasing the learning rate → Fixes high bias
Increasing the learning rate → Fixes high variance

Most of the time if you get a really low variance you'd have a high bias and the same is true for a low bias, for the best you'd want to balance the model bias variance tradeoff, so the model you use isnt too simple or too complex.

Large neural are low bias machines, if the network doesnt do well on the training data making the network bigger would improve it, if it doesnt do well on the dev set, adding more data would improve it.
A large neural network would do as good or better than a smaller better so long as the regularization is chosen well.
Adding regularization to a neural network in TensorFlow:
```py
model = Sequential([							#0.01 is the learning rate
		Dense(units=30, activation='relu', kernel_regularizer=L2(0.01), #hidden layer 1
		Dense(units=20, activation='relu', kernel_regularizer=L2(0.01), #hidden layer 2
		Dense(units=1, activation='linear', kernel_regularizer=L2(0.01) #output layer or layer 3
	])
```

[This lab](C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb) shows how bias and variance can be diagonsed.

## Machine Learning Development Process
### Iterative Loop
The best way to improve the accuracy of a model is to follow the iterative loop of machine learning development, to first choose and make the architecture and then train the model, then make diagnostics to see where you can improve the model and then loop back to architecture and repeat the steps.

![image](https://user-images.githubusercontent.com/121340570/231270129-f9d726ac-5141-4ef8-9d94-2cf59e415939.png)

### Error Analysis

One way to improve the accuracy of a system is to manually examine the misclassified outputs and categorizing them into common traits. For example if you had a system that would try to classify an email as spam or not, then you could classify them into many categories, such as, emails that try to steal passwords, emails that try to steal credit card info, emails that have deliberate misspellings. After doing that you can then check which of the missclassified emails are under which category, if you found that emails that try to steal passwords are 40% of the missclassifed emails then you can try to improve the accuracy for just those emails as they take a large portion of the misclassificated emails. This can be done by adding more emails that are trying to steal passwords to the data, or by adding more features that would help.

### Data Augmentation
Adding more data can usually improve the model and its accuracy, an easy way to do so is by performing data augmentation, by altering the data that you already have. 
For images you can for example distord the image, zoom in or zoom out, rotate or mirror the image, by doing so you would be adding new data from the existing data that was already there.

![Data augmentation](https://files.readme.io/ebfc7b6-Data_augmentation_methods_for_images.png)

Data augmentation can also be applied to speech recognition, you can to the original audio clip background sounds, or change the pitch of the audio clip, make the audio noisy. This can also be representative to how real audios could be.

### Transfer Learning
When the data you're working with to make a system is small then transfer learning can be very useful. By using an already trained model for a different task and then only training the output layer in the neural network. This can be very useful in image processing where the first layers basically just work to understand how to interpert the image. 
Transfer learning is a two step process: 
1. Supervised pretraining: training a model with a large dataset 
2. Fine tuning: taking the nueral network for the pretained model and fine tuning it to make it work for your model.

Fine tuning can be done two ways, either by training only the output layer parameters, or training all the parameters

## Error for Skewed Datasets
For problems when what you are trying to predict or classify is a small percentage of the entire dataset. For exmaple, trying to predict if a person has a rare disease, if 1% of people had that disease, then a system that would always predict the person does not have the disease would be 99% accurate. For problems with a skewed dataset it is better to use **percision** and **recall** to find the error of the model.

![image](https://user-images.githubusercontent.com/121340570/231283175-de3360eb-41f7-49fa-9324-c34d7b43d5de.png)

## Graded Lab 3
This [Graded Lab](C2_W3_Assignment.ipynb) covered how changing the number of layers and neurals in neural network affects its accuracy. 
