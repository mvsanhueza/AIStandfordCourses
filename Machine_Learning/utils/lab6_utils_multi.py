import numpy as np
import copy 
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 

def load_data_multi():
    data = np.loadtxt("data/houses.txt", delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y