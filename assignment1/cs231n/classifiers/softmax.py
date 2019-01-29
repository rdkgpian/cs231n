import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_classes=W.shape[1]
  prob_vec=np.zeros((num_classes,1))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  for i in range(num_train):
    scores=X[i].dot(W)
    scores=scores-np.max(scores)#Normalization trick to avoid the exp function shooting upto infinity
    denominator=np.sum(np.exp(scores))
    #numerator=np.exp(scores[y[i]])
    #loss+=-np.log((numerator)/denominator)
    prob_vec[y[i]]=np.exp(scores[y[i]])/denominator
    loss+=-np.log(prob_vec[y[i]])
    for j in range(num_classes):
        if(j!=y[i]):                  
            prob_vec[j]=np.exp(scores[j])/denominator
            dW[:,j]+=(prob_vec[j])*X[i]
        else:
            dW[:,j]+=(prob_vec[j]-1)*X[i]
        
        
        
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)
  dW/=num_train
  dW+=reg*W  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  prob_vec=np.zeros((num_train,W.shape[1]))
  scores=X.dot(W)
  scores-=np.max(scores)
  exp=np.exp(scores)
  denom=np.sum(exp,axis=1)
  #numer=scores[np.arange(num_train),y]
  prob_vec=exp/denom[:,np.newaxis]
  loss=np.sum(-np.log(prob_vec[np.arange(num_train),y]))
     
  temp=np.zeros_like(prob_vec)
  temp[np.arange(num_train),y]=1
  dW=X.T.dot(prob_vec-temp)
  
  
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)
  dW/=num_train
  dW+=reg*W
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

