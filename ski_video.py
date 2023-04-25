# -*- coding: utf-8 -*-

import numpy as np 
import cv2 
from matplotlib import pyplot as plt
pip install scikit-video
import skvideo.io  

"""# Load Video """
videodata = skvideo.io.vread("/content/ski_drop_low.mp4")

# Get dimensions 
number_of_frames, height, width, colors = videodata.shape

# Convert to greyscale 
grey_ski = skvideo.utils.rgb2gray(videodata)

height

width

"""# Create the matrix (of size M x N) whose columns are the vectorized frames """

frames = [] 

for frame in grey_ski:
  
    reshaped_frame = frame.reshape((540*960))   #Vectorize the frames by stacking the columns vertically 
    frames.append(reshaped_frame)               #it will add the vectorized into the matrix "frames" as rows 


Transpose_frames = np.array(frames).T

Transpose_frames.shape

"""# Do compact SVD """

u,s,vh = np.linalg.svd(Transpose_frames,full_matrices = False)

"""Produce the image of the original frame

I'm choosing frame 425 because the individual is visible at the bottom of the picture
"""

frame1=np.reshape(Transpose_frames[:,425],(height,width))
plt.figure(figsize=(6,6))
plt.title("Original")
plt.imshow(frame1,cmap='gray')

"""# Calculate energy of each singularvalue and the cumulative energy

"""

total_squared_sum_sigmas = sum(i*i for i in s)
energy_of_each_singular_value = [(i*i)/total_squared_sum_sigmas for i in s]
k = [(i+1) for i in range(len(s))]


sings = 0
cumulatives =[] 

for i in s:
  sings += i*i 
  cum = sings/total_squared_sum_sigmas
  cumulatives.append(cum)

"""# Function to construct the low-rank images that would help in deciding whats the best s 
Takes input: the number of singular values desired

Outputs the image of the low-rank approximated frame, image of the difference between the original and low-rank approximated frame, and the cumulative energy
"""

def low_rank_image_reconstruction(number_of_singularvalues):
  new_matrix = u[:,0:number_of_singularvalues].conj().T @ Transpose_frames
  f1 = u[:,0:number_of_singularvalues]@new_matrix

  # Create the energy 
  energy_used = cumulatives[number_of_singularvalues-1]*100



  #Extract the first frame 
  reduced_frame = np.reshape(f1[:,425],(height,width))

  fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,8))
  ax1.imshow(reduced_frame,cmap='gray')
  ax1.set_title("Low Rank Approximation (s="+str(number_of_singularvalues)+")")
  ax2.imshow(np.reshape(Transpose_frames[:,425]-f1[:,425],(height,width)),cmap='gray')
  ax2.set_title(" Difference ")

  fig.suptitle("Energy is: "+str(energy_used)+"%",y=0.7,fontsize=15)
  plt.show()

"""# Line Plot of Cumulative energy vs singularvalues k plt.plot(k,cumulatives)

"""

plt.plot(k,cumulatives)
plt.xlabel("K Singular Values ")
plt.ylabel("Energy")
plt.title("Plot of Cumulative Energy (Ski Drop Video)")

"""# Produce the low-rank approximated pictures and the difference """

low_rank_image_reconstruction(10)
low_rank_image_reconstruction(25)
low_rank_image_reconstruction(30)
low_rank_image_reconstruction(40)

"""# Create the low-rank approximated matrix (denoted as new_videodata)"""

# Specify the number of singular values
num_of_sings = 30


# The new low-rank approximated data
new_videodata = u[:,0:num_of_sings].conj().T @ Transpose_frames #Creates the low rank matrix

new_videodata.shape

# Create the matrices X and Y 
X = np.delete(new_videodata,453,axis=1)
Y = np.delete(new_videodata,0,axis=1)

"""# Create the DMD matrix A"""

A = Y @ np.linalg.pinv(X)

A.shape

# Get eigenvalues and eigenvectors of A
evalues, evectors = np.linalg.eig(A)

# Create omega: continuous time eigenvalues of A

omega =[]                 # set of transformed eigenvalues 

dt =8/number_of_frames    #duration of video/number of frames

for eigenvalue in evalues:
  omegak = np.log(eigenvalue)/dt
  omega.append(omegak)

omega = np.array(omega)

"""Plot the discrete time eigenvalues """

plt.scatter(evalues.real,evalues.imag)
plt.xlabel("Re($\mu$)")
plt.ylabel("Im($\mu$)")
plt.title("Plot of Discrete Time Eigenvalues $\mu$",fontsize=11)

"""Plot the continuous time eigenvalues 

(This will be used to determine a threshold to separate the fast and slow ones)
"""

plt.figure(figsize=(5,5))
plt.xlabel("Re($\omega$)")
plt.ylabel("Im($\omega$)")
plt.title("Plot of Continuous Time Eigenvalues $\omega$",fontsize=11)

# To determine the threshold, uncomment the following line. Zoom in according to your values
#plt.axis([-0.5,0.5, -1, 1])    

plt.scatter((omega).real,(omega).imag)

# Find coefficients of the eigendecomposition 

b = np.linalg.pinv(evectors) @ X[:,0]

b.shape

plt.figure(figsize=(10,10))

plt.imshow(np.reshape(u[:,0:num_of_sings]@new_videodata[:,425].real, (height, width)),cmap='gray') #since values in xdmd are real and imaginary part is really close to 0, I'm taking

"""Find index of the continuous time eigenvalues that fall below the threshold"""

slow_index = []
for i in range(len(omega)):
  if abs(omega[i])<0.5:
    slow_index.append(i)

"""Create a vector b corresponding to the coefficients of the slow modes that has zeros everywhere except at the position of the slow_index"""

b_slow = np.zeros(b.shape).astype(complex)

for i in slow_index:
  b_slow[i] = b[i]

np.array(slow_index).shape



"""The background (slow_x)"""

slow_x = np.zeros((num_of_sings,number_of_frames)).astype('complex')

for i in range(number_of_frames):
  for j in range(num_of_sings):
    if b_slow[j]!=0:
      slow_x[:,i] = slow_x[:,i]+(b[j]* np.exp(omega[j]*i))*evectors[:,j]

"""The foreground (x_sparse)"""

x_sparse = new_videodata - np.abs(slow_x)

"""Calculate the matrix R 

Since the above calculation may result in x_sparse having negative values in some of its elements, which would not make sense in terms of having negative pixel intensities. These residual negative values can be put into a n×m matrix R and then be added back into x_slow:
"""

new_r =np.zeros(x_sparse.shape)

for i in range(number_of_frames):
  for j in range(num_of_sings):
    if x_sparse[j,i].real <0:
      new_r[j,i] = x_sparse[j,i]

new_sparse = x_sparse - new_r

new_slow =np.abs(slow_x) + new_r

plt.imshow(np.reshape((u[:,0:num_of_sings]@ new_sparse[:,425]),(height, width)),cmap = 'gray')
plt.title("Foreground")

plt.imshow(np.reshape((u[:,0:num_of_sings]@ new_slow[:,425]),(height, width)),cmap = 'gray')
plt.title("Background")
