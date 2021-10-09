from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]
    for i in range(N):
      f_vec = X[i]@W
      f_vec -= np.max(f_vec)
      loss += -np.log( np.exp(f_vec[y[i]])/np.exp(f_vec).sum() )

      for j in range(C):
        if j == y[i]:
          continue
        dW[:,j] += np.exp(f_vec[j])/np.exp(f_vec).sum()*X[i]

      dW[:,y[i]] += -X[i] + np.exp(f_vec[y[i]])/np.exp(f_vec).sum()*X[i]

    loss /= N
    loss += reg*(W**2).sum()
    dW /= N
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]

    f_mat = X@W  # N*C

    f_mat -= np.max(f_mat,axis=1)[:,np.newaxis]

    exp_f_mat = np.exp(f_mat)
    softmax_scores = exp_f_mat / exp_f_mat.sum(1)[:,np.newaxis]  # 构造出这个矩阵是关键

    loss = -np.log(softmax_scores[range(N),y]).sum()

    softmax_scores[range(N),y] -= 1
    dW = X.T@softmax_scores
    # exp_f_cor_score = np.exp(f_mat[range(N),y])
    # exp_f_sum = np.exp(f_mat).sum(axis=1)
    # softmax_score = exp_f_cor_score / exp_f_sum 
    # loss = -np.log(softmax_score).sum()

    loss /= N
    loss += reg*(W**2).sum()
    dW /= N
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
