import numpy as np
import sklearn
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy.linalg
import sklearn.linear_model
import sklearn.preprocessing
from gradient_descent import computegrad,objective,bt_line_search,graddescent,objective_plot,plot_misclassification_error,compute_misclassification_error,generate_simulate_logistic



print('Generating simulate data by logstic regression ...')   
x,y=generate_simulate_logistic()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)
print('Spliting the data into train and test dataset......' )
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print('Standardlizing x_train and x_test......')
n_train = len(y_train)
n_test = len(y_test)
d = np.size(x, 1)
print('The size of train data is ',n_train)
print('The size of test data is',n_test)
print('The number of feature is ', d)

lr_cv = sklearn.linear_model.LogisticRegressionCV(penalty='l2', fit_intercept=False, tol=10e-8, max_iter=1000)
lr_cv.fit(x_train, y_train)
optimal_lambda = lr_cv.C_[0]
print('Finding the optimal lamba by cross validationn....')
print('The optimal lambda is:',optimal_lambda)

beta_init = np.zeros(d)
print('Initialing beta with 0......')
print('Beta_Init:',beta_init)

eta_init = 1/(scipy.linalg.eigh(1/len(y_train)*x_train.T.dot(x_train), eigvals=(d-1, d-1), eigvals_only=True)[0]+optimal_lambda)
print('Caculating the Initial Step Size......')
print('The Initial step Size is :', eta_init)

maxiter=1000
print('The max iterion is:',maxiter)

print('Caculating the optimal coefficients by Gradient Descent......')
betas_grad=graddescent(beta_init, optimal_lambda, eta_init, maxiter,x_train,y_train)

print('The optimal Coefficient is:',betas_grad[maxiter])
print('Generating Object value VS Iteration Plot......')
objective_plot(betas_grad,optimal_lambda,x_train,y_train)

print('Generating misclassification error VS Iteration Plot......')
plot_misclassification_error(betas_grad,x_train, y_train,title='Training set misclassification error')


