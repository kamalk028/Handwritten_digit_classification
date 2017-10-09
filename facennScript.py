'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
import datetime

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    s =  1/(1+np.exp(np.multiply(-1,z)))
    return s

def sigmoidDerivative(z):
    return z*(1-z)

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    #expanding the training data labels from 1D to 2D
    n = training_data.shape[0]
    training_label_expanded = np.zeros([n,n_class]) 
    pos = 0
    for k in range (training_label.shape[0]):
        pos = int(training_label[k])
        training_label_expanded[k][pos] = 1
                       
    for j in range(1):
        l0 = training_data
        x = np.hstack([l0, np.ones([l0.shape[0],1])])
        aj = np.dot(x,w1.T)
        l1 = sigmoid(aj)
        
        zj = np.hstack([l1, np.ones([l1.shape[0],1])])
        bl = np.dot(zj,w2.transpose())
        ol = sigmoid(bl)
        yl = training_label_expanded
        deltal =  ol - yl
        
        W2JP = np.dot(np.transpose(deltal),zj)
        grad_w2 = W2JP
        
        deltai = np.dot(deltal,w2)
        deltai = sigmoidDerivative(zj) * deltai
        deltai = np.delete(deltai,deltai.shape[1]-1,1) 
        W1JP = np.dot(np.transpose(deltai),x)
        grad_w1 = W1JP
        
        obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)    
        
        obj_grad = obj_grad / n
        
        #(1 − yil) ln(1 − oil)
        logErrR = (1 - yl) * (np.log(1-ol))
        # -yil ln oil
        logErrL =  yl * np.log(ol)
        logErr = np.sum(-1* (logErrL + logErrR))
        logErr = logErr / n
        
        
        RW1=np.sum(np.square(w1))
        RW2=np.sum(np.square(w2))
        RWSum = RW1 + RW2
        obj_val = (lambdaval / (2 * n)) * RWSum
        obj_val = logErr + obj_val
       
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
        
    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    l0 = data
    x = np.hstack([l0, np.ones([l0.shape[0],1])])
    aj = np.dot(x,w1.T)
    l1 = sigmoid(aj)
        
    zj = np.hstack([l1, np.ones([l1.shape[0],1])])
    bl = np.dot(zj,w2.transpose())
    ol = sigmoid(bl)
    
    labels=np.argmax(ol,axis=1)
    '''
    labels = np.array(labels)
    labels=np.reshape(labels,(-1,len(labels)))
    labels=np.transpose(labels)
    '''
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
t1 = datetime.datetime.now()
#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

selected_features = list(range(2376))

print('selected features: ')
#print(selected_features.shape)
print(selected_features)

obj = [selected_features, n_hidden, w1, w2, lambdaval]

pickle.dump(obj, open('params.pickle', 'wb'))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
t2 = datetime.datetime.now()
etime = t2 - t1
sec = etime.total_seconds()
time_taken = "Time Taken in Seconds : "+str(sec)
print(time_taken)