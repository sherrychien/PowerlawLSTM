# PowerLawLSTM
Implementing power law gated LSTM (pLSTM) on copy task with Pytorch.

## dataset.py
Generating training and testing datasets for the copy task. The T for the task could be flexibly adjusted to generate corresponding sequences for the task. 

## experiment.py
Running the copy task experiment including the model settings, training, validation and testing procedure etc.

## model.py
Including LSTM and pLSTM model classes for learning copy task.

## metric.py
Metric (calculating the accuracy in this case) for evaluating the model performance on the copy task


## CopyTask_run.ipynb
The main file for running the task using Jupyter notebook. Please install/import Pytorch and other required libraries along with the python files included in this zip file to run the copy task using LSTM/pLSTM.   
