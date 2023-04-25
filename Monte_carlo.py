# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r_sK2mU8gkl7xIbzNbSH7KulbBirWWb2
"""

import numpy as np 
import cv2 
from matplotlib import pyplot as plt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import time

pip install scikit-video

import skvideo.io  

videodata = skvideo.io.vread("/content/monte_carlo_low.mp4")  
number_of_frames, height, width, colors = videodata.shape
grey_monte = skvideo.utils.rgb2gray(videodata)
#print(videodata.shape)

# Create a 2D data matrix 
frames = [] 

for frame in grey_monte:

  #Vectorize the frames by stacking the columns vertically 
  reshaped_frame = frame.reshape((540*960))  

  frames.append(reshaped_frame)        #adds them to the matrix "frames" as rows 



# Transpose the matrix frames so that the each column represents a frame and dimension becomes space x time 
Transpose_frames = np.array(frames).T



# Compute the SVD 
u,s,vh = np.linalg.svd(Transpose_frames,full_matrices = False)

# Obtain the image of the first frame 

frame1 = np.reshape(Transpose_frames[:,0],(height,width))

plt.figure(figsize=(6,6))
plt.title("Original")
plt.imshow(frame1,cmap='gray')

s.shape

"""# *CREATE ENERGY LOW RANK *"""

total_squared_sum_sigmas = sum(i*i for i in s)
energy_of_each_singular_value = [(i*i)/total_squared_sum_sigmas for i in s]
k = [(i+1) for i in range(len(s))]


sings = 0
cumulatives =[] 

for i in s:
  sings += i*i 
  cum = sings/total_squared_sum_sigmas
  cumulatives.append(cum)

"""# fUNCTION TO RECONSTRUCT LOW RANK IMAGES ( FIND s )"""

#Function to construct the low rank images that would help in deciding whats the best s 
#Takes input as original_matrix u,s,vh and desired number of singular values that we want to use 

def low_rank_image_reconstruction(number_of_singularvalues):
  #new_matrix = u[:,0:number_of_singularvalues] @ np.diag(s[0:number_of_singularvalues]) @ vh[0:number_of_singularvalues,:] #Creates the low rank matrix 
  new_matrix = u[:,0:number_of_singularvalues].conj().T @ Transpose_frames
  f1 = u[:,0:number_of_singularvalues]@new_matrix

  # Create the energy 
  energy_used = cumulatives[number_of_singularvalues-1]*100
  #print("Energy is : ",energy_used,"%")



  #Extract the first frame 
  reduced_frame = np.reshape(f1[:,0],(height,width))

  fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,8))
  ax1.imshow(reduced_frame,cmap='gray')
  ax1.set_title("Low Rank Approximation (s="+str(number_of_singularvalues)+")")
  ax2.imshow(np.reshape(Transpose_frames[:,0]-f1[:,0],(height,width)),cmap='gray')
  ax2.set_title(" Difference ")

  fig.suptitle("Energy is: "+str(energy_used)+"%",y=0.7,fontsize=15)
  plt.show()

plt.plot(k,cumulatives)
plt.xlabel("K Singular Values ")
plt.ylabel("Energy")
plt.title("Plot of Cumulative Energy (M")

omega.shape

new_matrix = u[:,0:10].conj().T @ Transpose_frames
f1 = u[:,0:10]@new_matrix
f1.shape

new_matrix.shape

ewd = np.reshape(f1[:,0],(height,width))
plt.imshow(ewd,cmap='gray')

low_rank_image_reconstruction(50)
low_rank_image_reconstruction(100)
low_rank_image_reconstruction(150)
low_rank_image_reconstruction(170)

"""# CREATE REDUCED SVD using s """

num_of_sings = 170
new_videodata = u[:,0:num_of_sings].conj().T @ Transpose_frames #Creates the low rank matrix

new_videodata.shape

X = np.delete(new_videodata,378,axis=1)
Y = np.delete(new_videodata,0,axis=1)

X.shape

"""# DMD MATRIX A"""

A = Y @ np.linalg.pinv(X)

A.shape

"""Get eigenvalues/ vectors of A """

evalues, evectors = np.linalg.eig(A)

len(evalues)

mu = np.diag(evalues)

evectors.shape

plt.figure(figsize=(6,6))

plt.scatter(evalues.real,evalues.imag)
plt.xlabel("Re($\mu_i)$")
plt.ylabel("Im($\mu_i)$)")
plt.title("Plot of Discrete Time Eigenvalues $\mu_i$ (Monte Carlo Video)")
plt.axis([0, 1, -0.6, 0.6])

# Get the continuous eigenvalues by doing ln()

omega =[] # set of transformed eigenvalues 

dt =6/number_of_frames

for eigenvalue in evalues:
  omegak = np.log(eigenvalue)/dt
  omega.append(omegak)

omega = np.array(omega) #size (190, )

# Now, we need to determine the fast ones and slow ones 
# Plot the eigenvalues 
plt.figure(figsize=(4,4))
plt.xlabel("Re($\omega_i$)")
plt.ylabel("Im($\omega_i$)")
plt.title("Zoomed in", fontsize = 11)
#plt.axis([-1, 1, -0.007, 0.007])#zoom in 
plt.scatter((omega).real,(omega).imag)

# Find coefficients of the eigendecomposition 

b = np.linalg.pinv(evectors) @ X[:,0]

plt.figure(figsize=(10,10))

plt.imshow(np.reshape(transformed_xdmd1.real, (height, width)),cmap='gray') #since values in xdmd are real and imaginary part is really close to 0, I'm taking

"""# Create fast/slow modes 

"""

# Apply DMD

# Find the indexes of |omega|< 0.05
slow_index = []
for i in range(len(omega)):
  if abs(omega[i])<0.05:
    slow_index.append(i)

np.array(slow_index).shape

b_slow = np.zeros(b.shape).astype(complex)

for i in slow_index:
  b_slow[i] = b[i]

x = np.zeros((num_of_sings,number_of_frames)).astype('complex')

for j in range(number_of_frames):
  for i in range(num_of_sings):
    x[:,j] = x[:,j] + (b[i] * np.exp(evalues[i]* j ) )* evectors[:,i]

omega[1]

slow_x = np.zeros((num_of_sings,number_of_frames)).astype('complex')

for i in range(number_of_frames):
  for j in range(num_of_sings):
    if b_slow[j]!=0:
      slow_x[:,i] = slow_x[:,i]+(b[j]* np.exp(omega[j]*i))*evectors[:,j]

# once we get the coefficient of slow part 
# we need to create bp psi p u^t 

# CONSIDER T= 1
x_slow_mat = []
for i in range(number_of_frames):
  x_slow_try = evectors @ np.diag(np.exp( omega*i)) @ b_slow
  x_slow_mat.append(x_slow_try)

x_slow_mat = np.array(x_slow_mat).T

plt.imshow(np.reshape((u[:,0:num_of_sings]@x_slow_mat[:,0]).real,(height, width)),cmap = 'gray')

plt.imshow(np.reshape((u[:,0:num_of_sings]@slow_x[:,0]).real,(height, width)),cmap = 'gray')

slow_x

new_videodata.shape

x_sparse = new_videodata - np.abs(slow_x)

plt.imshow(np.reshape((u[:,0:num_of_sings]@x_sparse[:,0]).real,(height, width)),cmap = 'gray')

new_r =np.zeros(x_sparse.shape)

for i in range(number_of_frames):
  for j in range(num_of_sings):
    if x_sparse[j,i].real <0:
      new_r[j,i] = x_sparse[j,i]

new_r

new_sparse = x_sparse - new_r

new_sparse

new_sparse.shape

new_slow =slow_x + new_r # for some reason this gives me a better result but if I add it to new sparse, I don't get x

plt.imshow(np.reshape((u[:,0:num_of_sings]@ new_sparse[:,0]),(height, width)),cmap = 'gray')
plt.title("Foreground")

plt.imshow(np.reshape((u[:,0:num_of_sings]@new_slow[:,0]).real,(height, width)),cmap = 'gray')
plt.title("Background")

plt.imshow(np.reshape((u[:,0:num_of_sings]@x_sparse_try).real,(height, width)),cmap = 'gray')

np.array(b_slow).shape

new_slow = new_r + slow_x

new_slow1 = new_r + np.abs(slow_x)

plt.imshow(np.reshape((u[:,0:num_of_sings]@new_slow1[:,0]),(height, width)),cmap = 'gray')

new_slow1.dtype



def randomized(m, s,q,p):
  l = s+p
  n= m.shape[1]
  omega = np.random.normal(0, 1, size=(n, l))
  Z=m@omega 
  for k in range(q):
    Z= m@(m.T @Z)

  Q,R = np.linalg.qr(Z)

  return Q











