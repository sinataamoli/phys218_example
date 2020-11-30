#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
import time
import pint
import scipy.constants as cnst


# #### a) Schwarzchild radius of the Sun

#  <center> Schwarzchild radius: $r_{s}=\frac{2GM}{c^{2}}$

# In[72]:


M = 2e30    # Mass of the sun
G= cnst.G         # Gravitational const
c= cnst.c         # speed of light

def R(M):
    return 2*G*M/(c**2)

r=R(M) * u.meter

print(r)


# #### b) Matrix multiplication

# In[73]:


""" nested for loop """

# matrix A (3x4)
A = [[12, 7, 3, 3],
    [4, 5, 6, 9],
    [7, 8, 9, 1]]

# matrix B (4x3)
B = [[1,5,1],
    [2,8,3],
    [3,5,9],
    [3,4,1]]

# result is 3x4
result = [[0,0,0],
         [0,0,0],
         [0,0,0]]

# Matrix multiplication
for i in range(len(A)):
    for j in range(len(B[0])):
        for k in range(len(B)):
            result[i][j] += A[i][k] * B[k][j]

result


# In[74]:


""" list comprehension """
result_LC = [[sum(a*b for a,b in zip(A_row,B_col)) for B_col in zip(*B)] for A_row in A]
result_LC


# In[57]:


""" built-in numpy matrix multiplication """
np.matmul(A,B)


# #### c) speed test

# In[76]:


get_ipython().run_cell_magic('time', '', '\nC= [[0,0,10,13,0,0,0,0,1,0],\n    [0,0,1,20,1,15,0,0,0,0],\n    [0,1,0,1,0,1,32,0,21,0],\n    [0,0,1,11,0,41,1,0,0,0],\n    [0,1,0,19,0,1,65,1,0,0],\n    [0,0,1,0,1,78,1,12,1,0],\n    [12,0,0,1,0,1,0,76,0,1],\n    [0,13,0,0,1,0,0,13,1,0],\n    [1,0,14,0,0,1,0,1,43,1],\n    [0,0,0,16,0,0,1,0,1,65]]\n\ninv_C = np.linalg.inv(C)\n\nresult = [[0,0,0,0,0,0,0,0,0,0],\n          [0,0,0,0,0,0,0,0,0,0],\n          [0,0,0,0,0,0,0,0,0,0],\n          [0,0,0,0,0,0,0,0,0,0],\n          [0,0,0,0,0,0,0,0,0,0],\n          [0,0,0,0,0,0,0,0,0,0],\n          [0,0,0,0,0,0,0,0,0,0],\n          [0,0,0,0,0,0,0,0,0,0],\n          [0,0,0,0,0,0,0,0,0,0],\n          [0,0,0,0,0,0,0,0,0,0]]\nfor i in range(len(C)):\n    for j in range(len(inv_C[0])):\n        for k in range(len(inv_C)):\n            result[i][j] += C[i][k] * inv_C[k][j]')


# In[77]:


get_ipython().run_cell_magic('time', '', '\nC= [[0,0,10,13,0,0,0,0,1,0],\n    [0,0,1,20,1,15,0,0,0,0],\n    [0,1,0,1,0,1,32,0,21,0],\n    [0,0,1,11,0,41,1,0,0,0],\n    [0,1,0,19,0,1,65,1,0,0],\n    [0,0,1,0,1,78,1,12,1,0],\n    [12,0,0,1,0,1,0,76,0,1],\n    [0,13,0,0,1,0,0,13,1,0],\n    [1,0,14,0,0,1,0,1,43,1],\n    [0,0,0,16,0,0,1,0,1,65]]\n\ninv_C = np.linalg.inv(C)\n\nresult_LC = [[sum(a*b for a,b in zip(C_row, inv_C_col)) for inv_C_col in zip(*inv_C)] for C_row in C]')


# In[78]:


get_ipython().run_cell_magic('time', '', 'C= [[0,0,10,13,0,0,0,0,1,0],\n    [0,0,1,20,1,15,0,0,0,0],\n    [0,1,0,1,0,1,32,0,21,0],\n    [0,0,1,11,0,41,1,0,0,0],\n    [0,1,0,19,0,1,65,1,0,0],\n    [0,0,1,0,1,78,1,12,1,0],\n    [12,0,0,1,0,1,0,76,0,1],\n    [0,13,0,0,1,0,0,13,1,0],\n    [1,0,14,0,0,1,0,1,43,1],\n    [0,0,0,16,0,0,1,0,1,65]]\n\ninv_C = np.linalg.inv(C)\nresult3 = np.matmul(C,inv_C)')


# #### comparing the running times we can see that "numpy built-in function" is faster than the other routines
