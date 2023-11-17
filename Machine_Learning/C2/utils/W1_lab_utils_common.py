"""
lab_utils_common
   contains common routines and variable definitions
   used by all the labs in this week.
   by contrast, specific, large plotting routines will be in separate files
   and are generally imported into the week where they are used.
   those files will import this file
"""
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from ipywidgets import Output
from matplotlib.widgets import Button, CheckButtons

np.set_printoptions(precision=2)

dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0', dldarkblue =  '#0D5BDC')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; dldarkblue =  '#0D5BDC'
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

   z = np.clip(z, -500,500) #protect against overflow, no deja valores menores a -500 y mayores a 500 (los reemplaza)
   g = 1.0/(1.0+np.exp(-z))
   return g

