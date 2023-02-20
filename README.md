# Lotto Numbers Predictor

### Overview
This repository contains a lotto numbers predictor implemented using machine learning techniques (AI). Given sufficient publicly available data, this predictor can help remove the element of chance and provide a machine learning-based forecast of upcoming lottery numbers. 

While this project is for educational purposes only, it can be an enjoyable way to experiment with machine learning and neural networks while also learning about prediction methods.

### Requirements
All necessary files are included in the repository. The system is designed to work on both CPU and GPU depending on the configuration of your machine. The only prerequisites are TensorFlow and Pandas, which can be easily installed. The system is easy to understand and flexible, with the ability to change parameters.

### Usage
To use the lotto numbers predictor, simply download the repository and run the main.py file. The system will then process the data.csv file and generate an overview based on the performance of your computer. It will then attempt to predict 10 numbers in a row.

The main.py file contains various parameters that can be modified for training, including the training set, test set, number of epochs, batch size, learning rate, embedding dimensions, dropout rates, number of steps and features, hidden neurons, bidirectional flag, optimizer and loss function.

All results are logged in C:\Temp.

### Implementation

The lotto numbers predictor is implemented using TensorFlow and Keras. The system can automatically detect CUDA and switch from CPU to GPU if it is available. Other standard modules like NumPy and Pandas are also used. We recommend testing the system on both CPU and GPU using different CUDA versions through Anaconda to compare the performance.

### Result
So here is the result step by step.

First, publicly available data. Check

Then run main.py. Check

Wait until prediction is done. Check

The result of 7 numbers and the predicted success rate is 33%.

8 18 22 27 4 26 32

The real success of the model: 4 numbers

The winnings for a deposit of 1.5 Euros is 8 Euros.

Interested in your own model or forecast? Please feel free to contact me tomas.trnka.plc(slash)gmail.com