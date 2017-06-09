"""
"""


import numpy as np
import sklearn
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy.linalg
import sklearn.linear_model
import sklearn.preprocessing


def generate_simulate_logistic():
    np.random.seed(0)
    x1=np.random.normal(size=100)
    x2=np.random.normal(size=100)
    x3=np.random.normal(size=100)
    x4=np.random.normal(size=100)
    x5=np.random.normal(size=100)
    beta1=1
    beta2=2
    beta3=3
    beta4=4
    beta5=5
    print('Generating x1,x2,x3,x4,x5 by random normal number with size 100.....')
    print('Generating beta1,beta2,beta3,beta4,beta5 with value 1,2,3,4,5......')
    z=beta1*x1+beta2*x2+beta3*x3+beta4*x4+beta5*x5
    x=np.vstack((x1,x2,x3,x4,x5)).T
    pr = 1/(1+np.exp(-z)) 
    print('Caculating the probability by Logistic Regression......')
    y=(pr>0.5)*2-1
    print('Converting the probabiloity to 1/-1 with probability>0.5 or <0.5......')
    print('The Binary Indicator is',y)
    return x,y

def computegrad(beta, lambduh, x, y):
    """
	Compute the gradient respect to beta 
	:Beta: Input Beta value 
	:x: Input x 
	:y: Input y
	:lambduhL: Input lambduh
	:return: optimal beta
	"""
    yx = y[:,np.newaxis]*x
    denom = 1+np.exp(-yx.dot(beta))
    pena=2*lambduh*beta
    num=np.exp(-yx.dot(beta[:, np.newaxis]))
    regul=np.sum(-yx*num/denom[:, np.newaxis], axis=0)
    grad = 1/len(y)*regul+pena
    return grad

def objective(beta, lambduh, x, y):

	"""
	Find the value of objective function 
	:Beta: Input Beta value 
	:x: Input x 
	:y: Input y
	:lambduhL: Input lambduh
	:return: object value
	"""
	n=len(y)
	l2=lambduh*np.linalg.norm(beta)**2
	regul=np.sum(np.log(1+np.exp(-y*x.dot(beta))))
	return 1/n*regul+l2

def bt_line_search(beta, lambduh, x,y,eta=1, alpha=0.5, betaparam=0.8,maxiter=100):
    """
    Find the optimal step size for each new beta
    """
    grad_beta = computegrad(beta, lambduh, x, y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_eta = 0
    iter = 0
    while found_eta == 0 and iter < maxiter:
        if objective(beta - eta * grad_beta, lambduh, x=x, y=y) < objective(beta, lambduh, x=x, y=y) \
                - alpha * eta * norm_grad_beta ** 2:
            found_eta = 1
        elif iter == maxiter - 1:
            print('Max iterations reached')
        else:
            eta *= betaparam
            iter += 1
    return eta

def graddescent(beta_init, lambduh, eta_init, maxiter, x, y):
	"""
	Gradient Descent Alogorith to minimize the beta
	:Beta: An initial input Beta value 
	:lambduhL: Input optimal lambduh 
	:maxiter: maxiteration of algorithm
	:x: Input x_train
	:y: Input y_train 
	return: optimal beta values in an array 
	"""
	beta = beta_init
	grad_beta = computegrad(beta, lambduh, x, y)
	beta_vals = beta
    
	for i in range(maxiter):
		eta = bt_line_search(beta,lambduh,x,y,eta=eta_init)
		beta = beta - eta*grad_beta
		beta_vals = np.vstack((beta_vals, beta))
		grad_beta = computegrad(beta, lambduh, x,y)
		if i % 50 == 0:
			print('Processing Gradient descent iteration', i)
	return beta_vals

def objective_plot(betas_gd,lambduh, x, y):
	"""
	"""
	num_points = np.size(betas_gd, 0)
	objs_gd = np.zeros(num_points)
	for i in range(0, num_points):
		objs_gd[i] = objective(betas_gd[i, :], lambduh, x, y)
	plt.plot(range(1, num_points + 1), objs_gd, label='gradient descent')
	plt.xlabel('Iteration')
	plt.ylabel('Objective value')
	plt.title('Objective value vs. iteration when lambda='+str(lambduh))
	plt.show()



def plot_misclassification_error(betas_grad, x, y,title=''):
    niter = np.size(betas_grad, 0)
    error_grad = np.zeros(niter)
    error_fastgrad = np.zeros(niter)
    for i in range(niter):
        error_grad[i] = compute_misclassification_error(betas_grad[i, :], x, y)
    plt.plot(range(1, niter + 1), error_grad, label='gradient descent')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification error')
    if title:
        plt.title(title)
    plt.show()


def compute_misclassification_error(beta_opt, x, y):
    y_pred = 1/(1+np.exp(-x.dot(beta_opt))) > 0.5
    y_pred = y_pred*2 - 1  # Convert to +/- 1
    return np.mean(y_pred != y)



