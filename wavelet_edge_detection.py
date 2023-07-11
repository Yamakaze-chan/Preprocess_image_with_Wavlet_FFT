import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pywt
import scipy
from scipy import ndimage
from scipy import signal
from math import sqrt
import os
from os import walk


# Utility function for an Orthogonal wavelet filter set
# clone of the matlab function
# reference  https://github.com/gnattar/main/blob/master/universal/helper_funcs/wavelettool/orthfilt.m
def orthfilt(W):
    w = W/sum(W)
    # Associated filters 
    #LO_D = decomposition low-pass filter
    #HI_D = decomposition high-pass filter
    #LO_R = reconstruction low-pass filter
    #HI_R = reconstruction high-pass filter.
    
    LoF_R = sqrt(2)*w
    HiF_R = scipy.signal.qmf(LoF_R)
    HiF_D =  HiF_R[::-1]
    LoF_D =  LoF_R[::-1]

    return [LoF_D,HiF_D,LoF_R,HiF_R]

def resize (image):
  w, h = image.shape
  temp_max = max(w, h)
  scale_num = 1200.0/temp_max
  img = cv2.resize(image, (int(w*scale_num), int(h*scale_num)))
  return img

# Function to add gaussian noise to image
def gaussian_noise(image,sigma):
    row,col= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy
    
# Function to add Salt and pepper noise to image

def sp_noise(image,prob):
    
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


# Do wavelet_transform
def wavelet_transform(img,l,h):
    
    #Level-1
    app_h = scipy.ndimage.convolve1d(np.c_[np.zeros((img.shape[0],1)),img],l,axis=1,mode='constant')
    app_v = scipy.ndimage.convolve1d(app_h,l,axis=0,mode='constant')
    app = app_v[:,1:]
    
    H_h = scipy.ndimage.convolve1d(np.c_[np.zeros((img.shape[0],1)),img],h,axis=1,mode='constant')
    H_v = scipy.ndimage.convolve1d(H_h,l,axis=0,mode='constant')
    H = H_v[:,1:]
    
    V_h = scipy.ndimage.convolve1d(np.c_[np.zeros((img.shape[0],1)),img],l,axis=1,mode='constant')
    V_v = scipy.ndimage.convolve1d(V_h,h,axis=0,mode='constant')
    V = V_v[:,1:]
    
    D_h = scipy.ndimage.convolve1d(np.c_[np.zeros((img.shape[0],1)),img],h,axis=1,mode='constant')
    D_v = scipy.ndimage.convolve1d(D_h,h,axis=0,mode='constant')
    D = D_v[:,1:]
    
    scale_down = int(img.shape[0]/2)
    apps = cv2.resize(app, (scale_down,scale_down))
    Hs = cv2.resize(H,(scale_down,scale_down))
    Vs = cv2.resize(V, (scale_down,scale_down))
    Ds = cv2.resize(D, (scale_down,scale_down))
    
    # Level-2
    
    app1_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps.shape[0],1)),apps],l,axis=1,mode='constant')
    app1_v = scipy.ndimage.convolve1d(app_h,l,axis=0,mode='constant')
    app1 = app1_v[:,1:]
    
    H1_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps.shape[0],1)),apps],h,axis=1,mode='constant')
    H1_v = scipy.ndimage.convolve1d(H1_h,l,axis=0,mode='constant')
    H1 = H1_v[:,1:]
    
    V1_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps.shape[0],1)),apps],l,axis=1,mode='constant')
    V1_v = scipy.ndimage.convolve1d(V1_h,h,axis=0,mode='constant')
    V1 = V1_v[:,1:]
    
    D1_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps.shape[0],1)),apps],h,axis=1,mode='constant')
    D1_v = scipy.ndimage.convolve1d(D1_h,h,axis=0,mode='constant')
    D1 = D1_v[:,1:]
    
    scale_down = int(apps.shape[0]/2)
    apps1 = cv2.resize(app1, (scale_down,scale_down))
    H1s = cv2.resize(H1,(scale_down,scale_down))
    V1s = cv2.resize(V1, (scale_down,scale_down))
    D1s = cv2.resize(D1, (scale_down,scale_down))
    
    # Level 3
    
    app2_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps1.shape[0],1)),apps1],l,axis=1,mode='constant')
    app2_v = scipy.ndimage.convolve1d(app2_h,l,axis=0,mode='constant')
    app2 = app2_v[:,1:]
    
    H2_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps1.shape[0],1)),apps1],h,axis=1,mode='constant')
    H2_v = scipy.ndimage.convolve1d(H2_h,l,axis=0,mode='constant')
    H2 = H2_v[:,1:]
    
    V2_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps1.shape[0],1)),apps1],l,axis=1,mode='constant')
    V2_v = scipy.ndimage.convolve1d(V2_h,h,axis=0,mode='constant')
    V2 = V2_v[:,1:]
    
    D2_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps1.shape[0],1)),apps1],h,axis=1,mode='constant')
    D2_v = scipy.ndimage.convolve1d(D2_h,h,axis=0,mode='constant')
    D2 = D2_v[:,1:]
    
    scale_down = int(apps1.shape[0]/2)
    apps2 = cv2.resize(app2, (scale_down,scale_down))
    H2s = cv2.resize(H2,(scale_down,scale_down))
    V2s = cv2.resize(V2, (scale_down,scale_down))
    D2s = cv2.resize(D2, (scale_down,scale_down))
    
    # Level 4
    
    app3_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps2.shape[0],1)),apps2],l,axis=1,mode='constant')
    app3_v = scipy.ndimage.convolve1d(app3_h,l,axis=0,mode='constant')
    app3 = app3_v[:,1:]
    
    H3_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps2.shape[0],1)),apps2],h,axis=1,mode='constant')
    H3_v = scipy.ndimage.convolve1d(H3_h,l,axis=0,mode='constant')
    H3 = H3_v[:,1:]
    
    V3_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps2.shape[0],1)),apps2],l,axis=1,mode='constant')
    V3_v = scipy.ndimage.convolve1d(V3_h,h,axis=0,mode='constant')
    V3 = V3_v[:,1:]
    
    D3_h = scipy.ndimage.convolve1d(np.c_[np.zeros((apps2.shape[0],1)),apps2],h,axis=1,mode='constant')
    D3_v = scipy.ndimage.convolve1d(D3_h,h,axis=0,mode='constant')
    D3 = D3_v[:,1:]
    
    scale_down = int(apps2.shape[0]/2)
    
    apps3 = cv2.resize(app3, (scale_down,scale_down))
    H3s = cv2.resize(H3,(scale_down,scale_down))
    V3s = cv2.resize(V3, (scale_down,scale_down))
    D3s = cv2.resize(D3, (scale_down,scale_down))
    
    
    return ([apps,Hs,Vs,Ds,apps1,H1s,V1s,D1s,apps2,H2s,V2s,D2s,apps3,H3s,V3s,D3s])
    

# utility function to apply wavelet transform function to all the images
def wavelet_allImage(img,imggnoise,imgspnoise,l1,h1,scale,w,h):
    [app,H,V,D,app1,H1,V1,D1,app2,H2,V2,D2,app3,H3,D3,V3] = wavelet_transform(img,l1,h1)
    [appg,Hg,Vg,Dg,appg1,Hg1,Vg1,Dg1,appg2,Hg2,Vg2,Dg2,appg3,Hg3,Dg3,Vg3] = wavelet_transform(imggnoise,l1,h1)
    [appi,Hi,Vi,Di,appi1,Hi1,Vi1,Di1,appi2,Hi2,Vi2,Di2,appi3,Hi3,Di3,Vi3] = wavelet_transform(imgspnoise,l1,h1)
    
    
    return ([H,V,Hg,Vg,Hi,Vi])

# utility function to carry out thresholding
def thresholding(HH,VV,Edge,H,H2,V,V2,w,h):
    #angle = np.zeros([256,256])
    #output = np.zeros([256,256])
    #print(w,h)
    if w > 600: 
      w = 600
    if h > 600:
      h = 600
    angle = np.zeros([w,h])
    output = np.zeros([w,h])
    for i in range(0,int(w)-1):
        for j in range(0,int(h)-1):
            p = np.arctan(abs(VV[i][j]/HH[i][j]) * 180/np.pi)
            angle[i][j] = p
            
    edge_array = np.zeros([w,h])
    Gradient= Edge
    
    for i in range(1,int(w-1)):
        for j in range(1,int(h-1)):
            if ((angle[i][j]>=(-22.5)) and (angle[i][j]<=(22.5)) or (angle[i][j]>=(180-22.5)) \
                and (angle[i][j]<=(180+22.5))):
                if Gradient[i][j] > Gradient[i+1][j] and Gradient[i][j]>Gradient[i-1][j]:
                    edge_array[i][j] = Gradient[i][j]
            elif ((angle[i][j]>=(90-22.5)) and angle[i][j] <= (90+22.5) or angle[i][j]>=(270-22.5)\
                 and angle[i][j]<= 270+22.5):
                if Gradient[i][j] > Gradient[i][j+1] and Gradient[i][j]>Gradient[i][j-1]:
                    edge_array[i][j] = Gradient[i][j]
            elif ((angle[i][j]>=(45-22.5)) and angle[i][j]<=(45+22.5) or angle[i][j]>=(225-22.5) \
                 and angle[i][j] <=(225+22.5)):
                if Gradient[i][j]> Gradient[i+1][j+1] and Gradient[i][j]>Gradient[i-1][j-1]:
                    edge_array[i][j] = Gradient[i][j]
            else:
                if Gradient[i][j]>Gradient[i+1][j-1] and Gradient[i][j] >Gradient[i-1][j+1]:
                    edge_array[i][j] = Gradient[i][j]
                    
    
    aaa = np.max(edge_array)
    edge_array = edge_array/aaa
    
    for i in range(0,w):
        for j in range(0,h):
            if(edge_array[i][j]>0.2):
                output[i][j]=1
            else:
                output[i][j]=0
    scale_up = output.shape[0]
    resized_output = cv2.resize(output,(scale_up,scale_up))
    return resized_output
    
    
# utility function to carry out scale multiplication for edge detection
def scale_multiplication(H,H2,V,V2,Hg,Hg2,Vg,Vg2,Hi,Hi2,Vi,Vi2,tx,w,h):
    HH = np.multiply(H,H2)
    VV = np.multiply(V,V2)
    Edge = np.sqrt(abs(HH+VV))
    
    HHg = np.multiply(Hg,Hg2)
    VVg = np.multiply(Vg,Vg2)
    Edgeg = np.sqrt(abs(HHg+VVg))
    
    HHi = np.multiply(Hi,Hi2)
    VVi = np.multiply(Vi,Vi2)
    Edgei = np.sqrt(abs(HHi+VVi))
    
    fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(15, 10),
                        sharex=True, sharey=True)
    
    output_image = thresholding(HH,VV,Edge,H,H2,V,V2,w,h)
    ax1[0].imshow(output_image)
    ax1[0].set_title('Edge of Original Image ')
    output_imageG = thresholding(HHg,VVg,Edgeg,Hg,Hg2,Vg,Vg2,w,h)
    ax1[1].imshow(output_imageG)
    ax1[1].set_title('Edge of Gaussian Noise Corrupted Image ')
    output_imageSP = thresholding(HHi,VVi,Edgei,Hi,Hi2,Vi,Vi2,w,h)
    ax1[2].imshow(output_imageSP)
    ax1[2].set_title('Edge of S&P Noise Corrupted Image ')
    fig1.suptitle("____________________________________________"+tx+"____________________________________________", fontsize=20)
    
    
#
start_time = time.time()
path = "6933_01.jpg"
#path = "/content/FbzEfxYaAAA0Wj7.jfif"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float')
img1 = resize(img) 
img1gnoise = gaussian_noise(img1,20)
img1spnoise = sp_noise(img1,0.005)
shape_of_img = img1.shape
#print(shape_of_img)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 10),
                       sharex=True, sharey=True)
ax[0].imshow(img1, cmap = 'gray')
ax[0].set_title("Original Image #")
ax[1].imshow(img1gnoise, cmap = 'gray')
ax[1].set_title("White Noise corrupted image")
ax[2].imshow(img1spnoise, cmap = 'gray')
ax[2].set_title("Salt and Pepper noise corrupted image")

# Wavelet functions from pywt library
# haar
wavelet = pywt.Wavelet('haar')
[phi1, psi1, x1] = wavelet.wavefun(level=2)  # level 2
[phi2,psi2,x2] = wavelet.wavefun(level=3)  # level 3

# calculate filter parameters
# l1,l2 = decomposition LPF
# h1,h2 = decomposition HPF
[l1,h1,lr1,hr1] = orthfilt(phi1)
[l2,h2,lr2,hr2] = orthfilt(phi2)


# Edge detection using haar
[H,V,Hg,Vg,Hi,Vi] = wavelet_allImage(img1,img1gnoise,img1spnoise,l1,h1,' scale 1',shape_of_img[0], shape_of_img[1])
[H2,V2,Hg2,Vg2,Hi2,Vi2] = wavelet_allImage(img1,img1gnoise,img1spnoise,l2,h2,' scale 2',shape_of_img[0], shape_of_img[1])

scale_multiplication(H,H2,V,V2,Hg,Hg2,Vg,Vg2,Hi,Hi2,Vi,Vi2, 'haar wavelet ',shape_of_img[0], shape_of_img[1])


"""
#Edge Detection using db2
wavelet = pywt.Wavelet('db5')
[phi1, psi1, x1] = wavelet.wavefun(level=2)  # level 2
[phi2,psi2,x2] = wavelet.wavefun(level=3)  # level 3

[l1,h1,lr1,hr1] = orthfilt(phi1)
[l2,h2,lr2,hr2] = orthfilt(phi2)

# Edge detection using db2
[H,V,Hg,Vg,Hi,Vi] = wavelet_allImage(img1,img1gnoise,img1spnoise,l1,h1,' scale 1',shape_of_img[0], shape_of_img[1])
[H2,V2,Hg2,Vg2,Hi2,Vi2] = wavelet_allImage(img1,img1gnoise,img1spnoise,l2,h2,' scale 2',shape_of_img[0], shape_of_img[1])

scale_multiplication(H,H2,V,V2,Hg,Hg2,Vg,Vg2,Hi,Hi2,Vi,Vi2,' db5 wavelet ',shape_of_img[0], shape_of_img[1])


# Edge Detection using symm2
wavelet = pywt.Wavelet('coif3')
[phi1, psi1, x1] = wavelet.wavefun(level=2)  # level 2
[phi2,psi2,x2] = wavelet.wavefun(level=3)  # level 3

[l1,h1,lr1,hr1] = orthfilt(phi1)
[l2,h2,lr2,hr2] = orthfilt(phi2)

# Edge detection using haar
[H,V,Hg,Vg,Hi,Vi] = wavelet_allImage(img1,img1gnoise,img1spnoise,l1,h1,' scale 1',shape_of_img[0], shape_of_img[1])
[H2,V2,Hg2,Vg2,Hi2,Vi2] = wavelet_allImage(img1,img1gnoise,img1spnoise,l2,h2,' scale 2',shape_of_img[0], shape_of_img[1])

scale_multiplication(H,H2,V,V2,Hg,Hg2,Vg,Vg2,Hi,Hi2,Vi,Vi2, ' coif3 wavelet ',shape_of_img[0], shape_of_img[1])
"""

print("Execution took " + str(time.time()-start_time) + " seconds")
