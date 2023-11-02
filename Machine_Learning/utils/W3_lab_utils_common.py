import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from ipywidgets import Output

np.set_printoptions(precision=2)

dlc = dict(
    dlblue="#0096ff",
    dlorange="#FF9300",
    dldarkred="#C00000",
    dlmagenta="#FF40FF",
    dlpurple="#7030A0",
)
dlblue = "#0096ff"
dlorange = "#FF9300"
dldarkred = "#C00000"
dlmagenta = "#FF40FF"
dlpurple = "#7030A0"
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]


def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     g : array_like
         sigmoid(z)
    """

    z = np.clip(z, -500, 500)
    g = 1.0 / (1.0 + np.exp(-z))

    return g


##########################################################
# Regression Routines
##########################################################


def predict_logistic(X, w, b):
    """performs prediction"""
    return sigmoid(X @ b + b)


def predict_linear(X, w, b):
    """preforms prediction"""
    return X @ w + b

def log_1pexp(x, maxmium=20):
    ''' approximate log(1+exp^x)
        https://stats.stackexchange.com/questions/475589/numerical-computation-of-cross-entropy-in-practice
    Args:
    x   : (ndarray Shape (n,1) or (n,)  input
    out : (ndarray Shape matches x      output ~= np.log(1+exp(x))
    '''
    out = np.zeros_like(x,dtype=float)
    i = x <= maxmium
    ni = np.logical_not(i)

    out[i] = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]

    return out

def compute_cost_matrix(X,y,w,b,logistic=False, lambda_=0, safe = True):
    """
    Computes the cost using  using matrices
    Args:
      X : (ndarray, Shape (m,n))          matrix of examples
      y : (ndarray  Shape (m,) or (m,1))  target value of each example
      w : (ndarray  Shape (n,) or (n,1))  Values of parameter(s) of the model
      b : (scalar )                       Values of parameter of the model
      verbose : (Boolean) If true, print out intermediate value f_wb
    Returns:
      total_cost: (scalar)                cost

    """

    m = X.shape[0]
    y = y.reshape(-1,1) #Ensure 2d
    w = w.reshape(-1,1)
    if logistic:
        if safe:
            z = X @ w + b       #(m,n)*(n,1) = (m,1)
            cost = -(y*z) + log_1pexp(z) 
            cost = np.sum(cost) / m
        else:
            f = sigmoid(X @ w + b)
            cost = (1/m)*(np.dot(-y.T, np.log(f)) - np.dot((1-y).T, np.log(1-f)))  #(1,m)(m,1)=(1,1)
            cost = cost[0,0]
    else:
        f = X @ w + b
        cost = (1/(w*m))*np.sum((f-y)**2)
    
    reg_cost = (lambda_ / (2*m))* np.sum(w**2)

    total_cost = cost + reg_cost

    return total_cost

def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc="best"):
    """plots logistic data with two axis"""
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(
        -1,
    )  # work with 1D or 2D y vectors
    neg = neg.reshape(
        -1,
    )

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker="x", s=s, c="red", label=pos_label)
    ax.scatter(
        X[neg, 0],
        X[neg, 1],
        marker="o",
        s=s,
        label=neg_label,
        facecolors="none",
        edgecolors=dlblue,
        lw=3,
    )
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False


# Draws a theshold at 0.5
def draw_vthresh(ax, x):
    """draws a threshold"""
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.fill_between([xlim[0], x], [ylim[1], ylim[1]], alpha=0.2, color=dlblue)
    ax.fill_between([x, xlim[1]], [ylim[1], ylim[1]], alpha=0.2, color=dldarkred)
    ax.annotate(
        "z >= 0",
        xy=[x, 0.5],
        xycoords="data",
        xytext=[30, 5],
        textcoords="offset points",
    )
    d = FancyArrowPatch(
        posA=(x, 0.5),
        posB=(x + 3, 0.5),
        color=dldarkred,
        arrowstyle="simple, head_width=5, head_length=10, tail_width=0.0",
    )
    ax.add_artist(d)
    ax.annotate(
        "z < 0",
        xy=[x, 0.5],
        xycoords="data",
        xytext=[-50, 5],
        textcoords="offset points",
        ha="left",
    )
    f = FancyArrowPatch(
        posA=(x, 0.5),
        posB=(x - 3, 0.5),
        color=dlblue,
        arrowstyle="simple, head_width=5, head_length=10, tail_width=0.0",
    )
    ax.add_artist(f)
