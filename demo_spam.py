import numpy as np
import sklearn
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy.linalg
import sklearn.linear_model
import sklearn.preprocessing
from gradient_descent import computegrad,objective,bt_line_search,graddescent,objective_plot,plot_misclassification_error,compute_misclassification_error

spam = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
print('Loading data from: https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data .....')

test_indicator = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest', sep=' ',
                               header=None)
test_indicator = np.array(test_indicator).T[0]
x = np.asarray(spam)[:, 0:-1]
y = np.asarray(spam)[:, -1]*2 - 1  
print('Initialing value to x and y....')

x_train = x[test_indicator == 0,:]
x_test = x[test_indicator == 1,:]
y_train = y[test_indicator == 0]
y_test = y[test_indicator == 1]
print('Spliting data to training and test dataset, x_train,x_test, y_train,y_test....')

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print('Standardlizing x_train and x_test....')

n_train = len(y_train)
n_test = len(y_test)
d = np.size(x, 1)
print('The size of x_trainis dataset=',n_train)
print('The size of x_test dataset =',n_test)
print('The features of x_train =',d)

lr_cv = sklearn.linear_model.LogisticRegressionCV(penalty='l2', fit_intercept=False, tol=10e-8, max_iter=1000)
lr_cv.fit(x_train, y_train)
optimal_lambda = lr_cv.C_[0]
print('Finding optimal lambda by sk-leran cross validation...')
print('The Optimal lambda=', optimal_lambda)

beta_init = np.zeros(d)
print('Initialing beta initial...')
eta_init = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals=(d-1, d-1), eigvals_only=True)[0]+optimal_lambda)
print('Finding the optimal initial step size...')
print('Initial step size =',eta_init)
maxiter=300
print('Setting maxiteration =:',maxiter)
print('Finding Optimal beta by using gradient descent....')
betas_grad = graddescent(beta_init, optimal_lambda, eta_init, maxiter,x_train,y_train)
print('The optimal Coefficient is:',betas_grad[maxiter])


print('Generating Object value VS Iteration Plot....')
objective_plot(betas_grad,optimal_lambda,x_train,y_train)

print('Generating misclassification error VS Iteration Plot......')
plot_misclassification_error(betas_grad,x_train, y_train,title='Training set misclassification error')