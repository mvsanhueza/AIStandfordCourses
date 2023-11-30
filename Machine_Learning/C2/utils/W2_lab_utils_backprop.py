from sympy import *
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

def between(a,b,x):
    '''determine if a point x is between a and b. a may be reater or less than b'''
    if a > b:
        return b <= x <= a
    if b > a:
        return a <= x <= b

def near(pt, alist, dist=15):
    for a in alist:
        x, y = a.a0.get_position() #(bot left, bot right) data coords, not relative
        x = x -5
        y = y + 2.5
        if 0 < (pt[0] - x) < 25 and 0 < (y - pt[1]) < 25:
            return (True, a)
    return (False, None)

def inboxes(pt, boxlist):
    '''returns true if pt is within one of the boxes in boxlist'''
    for b in boxlist:
        if b.inbox(pt):
            return (True, b)
    return(False, None)

class avalue():
    ''' one of the values on the figure that can be filled in '''
    def __init__(self, value, pt, cl):
        self.value = value
        self.cl = cl  # color
        self.pt = pt  # print
    
    def add_anote(self, ax):
        self.ax = ax
        self.ao = self.ax.annote("?", self.pt, c=self.cl, fontsize='x-small')
    
class astring():
    ''' a string that can be set visible or invisible '''
    def __init__(self, ax, string, pt, cl):
        self.string = string
        self.cl = cl
        self.pt = pt
        self.ax = ax
        self.ao = self.ax.annotate(self.string, self.pt, c="white", fontsize='x-small')

    def astring_visible(self):
        self.ao.set_color(self.cl)
    
    def astring_invisible(self):
        self.ao.set_color("white")

class abox():
    ''' one of the boxs in the graph that has a value '''
    def __init__(self, ax, value, left, bottom, right, top, anpt, cl, adj_anote_obj):
        self.ax = ax
        self.value = value # correct value for annotation
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.anpt = anpt  # x,y where expression should be listed
        self.cl = cl
        self.ao = self.ax.annotate("?", self.anpt, c=self.cl, fontsize='x-small')
        self.astr = adj_anote_obj # 2ndary text for marking edges or none
    
    def inbox(self, pt):
        ''' true if point is within the box '''
        x, y = pt
        isbetween = between(self.top, self.bottom, y) and between(self.left, self.right, x)
        return isbetween
    
    def update_val(self, value, cl=None):
        self.ao.set_text(value)
        if cl:
            self.ao.set_c(cl)
        else:
            self.ao.set_c(self.cl)
    
    def show_secondary(self):
        if self.astr: # if there is a 2ndary tet of text
            self.astr.ao_set_c("green")
    
    def clear_secondary(self):
        if self.astr: # if there is a 2ndary tet of text
            self.astr.ao.set_c("white")
    

class plt_network():
    def __init__(self, fn, image, out=None):
        self.out = out #debug
        img = plt.imread(image)
        self.fig, self.ax = plt.subplots(figsize=self.sizefig(img))
        boxes = fn(self.ax)
        self.boxes = boxes
        widgvis(self.fig)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.ax.imshow(img)
        self.fig.text(0.1, 0.9, "Click in boxes to fill in values.")
        self.glist = [] # place to stash global things
        self.san = [] # selected annotation

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.axreveal = plt.axes([0.55, 0.02, 0.15, 0.075]) #[left, bottom, width, height]
        self.axhide = plt.axes([0.76, 0.02, 0.15, 0.075])
        self.breveal = Button(self.axreveal, 'Reveal All')
        self.breveal.on_clicked(self.reveal_values)
        self.bhide = Button(self.axhide, 'Hide All')
        self.bhide.on_clicked(self.hide_values)
    
    def sizefig(self, img):
        iy, ix, iz = np.shape(img)
        if 10/5 < ix/iy: #if x is the limiting size
            figx = 10
            figy = figx * iy / ix
        else:
            figy = 5
            figx = figy * ix / iy
        
        return (figx, figy)

    