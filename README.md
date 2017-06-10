# Gradient Descent for Logistic Regression 

l2 regularized logistic regression is one of common regularized regressions. 
, I use Gradient Descent to find the optimal coefficients. 

![alt text]([https://imgbb.com/][img]https://image.ibb.co/bLvX0v/1.png)

Gradient Descent is also called as Steepest Descent, it is fisrt order iterative optimization algorithm. It work to find local minimum of a function. The Alogrithm is described as follow: 

![alt text](screenshots/1 "Description goes here")

## Data

- 1: Simple Simulate Dataset

- [2: Spam DataSet](https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data)



## Software dependencies and license information
#### Programming language: 

- Python version 3.0 and above 

#### Python packages needed:

- pandas
- NumPy
- sklearn
- scipy 
- matplotlib 

#### Demo and Function : 

There are three deomo files and one Function file: 

demo_simulate.py allows a user to launch the method on a simple simulated dataset, visualize the training process, and print the performance. Simulated Data Already gernerated in demo file .

demo_spam.py allows a user to launch the method on a spam data, visualize the training process, and print the performance. Spam Data path already included in demo file  .

comparison.py allows a user to compare result of simulate data  with scikit-learn. Data source and information is included in the demo file.

gradient_descent.py includes all the function used for gradient_descent Algorthmns.

