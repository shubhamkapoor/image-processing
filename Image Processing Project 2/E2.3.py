# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:24:07 2017

@author: Molli
"""

import numpy as np
import scipy.misc as msc

a = []
b = []
a_ = []

def readIntensityImage(filename):
    f = msc.imread(filename, flatten=True).astype('float')
    return f
    
def getCoefficients(sigma):
    alpha_1 =  1.6800
    alpha_2=  -0.6803
    beta_1=  3.7350
    beta_2=  -0.2598
    gamma_1=  1.7830
    gamma_2=  1.7230
    omega_1=  0.6318
    omega_2=  1.9970
    
     
    a0 =  alpha_1 + alpha_2
    a1 =  (np.exp(-gamma_2 / sigma) * ((beta_2 * np.sin(omega_2/sigma)) -  ((alpha_2 + (2.0 * alpha_1)) * np.cos(omega_2/ sigma)))) + (np.exp(-gamma_1 / sigma) * ((beta_1 * np.sin(omega_1/sigma)) -  ((alpha_1 + (2.0 * alpha_2)) * np.cos(omega_1/ sigma))))
    #mayesha: a2 =  (((2 * np.exp(-((gamma_1 + gamma_2) / sigma))) * (((alpha_1 + alpha_2) * np.cos(omega_2/ sigma) * np.cos(omega_1/ sigma)) - (np.cos(omega_2/ sigma) * beta_1 * np.sin(omega_1/sigma)) - (np.cos(omega_1/ sigma) * beta_2 * np.sin(omega_2/sigma))))) + (alpha_2 * np.exp((- 2.0 * gamma_1) / sigma)) + (alpha_1 * np.exp((- 2.0 * gamma_2) / sigma))
    a2 = (2 * np.exp(-((gamma_1 + gamma_2) / sigma))) * (((alpha_1 + alpha_2) * np.cos(omega_2/sigma) * np.cos(omega_1/sigma)) - (np.cos(omega_2/sigma)*beta_1*np.sin(omega_1/sigma)) - (np.cos(omega_1/sigma)*beta_2*np.sin(omega_2/sigma))) + (alpha_2 * np.exp(-2*gamma_1/sigma)) + (alpha_1*np.exp(-2*gamma_2/sigma))
    #mayesha: a3 = (np.exp(-((gamma_2 + (2.0 * gamma_1))/ sigma)) * ((beta_2 * np.sin(omega_2/sigma)) - (alpha_2 * np.cos(omega_2/ sigma)))) + (np.exp(-((gamma_1 + (2.0 * gamma_2)) / sigma)) * ((beta_1 * np.sin(omega_1/sigma)) - (alpha_1 * np.cos(omega_1/ sigma))))
    a3 = (np.exp(-(gamma_2+(2*gamma_1))/sigma) * ((beta_2*np.sin(omega_2/sigma)) - (alpha_2*np.cos(omega_2/sigma))) ) + (np.exp(-(gamma_1+(2*gamma_2))/sigma) * ((beta_1*np.sin(omega_1/sigma)) - (alpha_1*np.cos(omega_1/sigma))) )
    
    a.append(a0)
    a.append(a1)
    a.append(a2)
    a.append(a3)
    
    b1=   (-2.0 * np.exp(-gamma_2 / sigma) * np.cos(omega_2/ sigma))-(2.0 * np.exp(-gamma_1 / sigma) * np.cos(omega_1/ sigma))
    b2=   (4.0 * np.cos(omega_2/ sigma) * np.cos(omega_1/ sigma) * np.exp(-(gamma_2+gamma_1)/sigma)) + np.exp((-2.0 * gamma_2) / sigma) + np.exp((-2.0 * gamma_1) / sigma)
    b3 =  (-2.0 * np.cos(omega_1/ sigma) * np.exp(-(gamma_1+(2*gamma_2))/sigma))-(2.0 * np.cos(omega_2/ sigma) * np.exp(-(gamma_2+(2*gamma_1))/sigma))
    b4 =  np.exp(-((2*gamma_1)+(2*gamma_2))/sigma)
    
    b.append(b1)
    b.append(b2)
    b.append(b3)
    b.append(b4)
    
    a_1 = a1 - (b1 * a0)
    a_2 = a2 - (b2 * a0)
    a_3 = a3 - (b3 * a0)
    a_4 = - (b4 * a0)
    
    a_.append(a_1)
    a_.append(a_2)
    a_.append(a_3)
    a_.append(a_4)
    
    return (a,b,a_)
    
def recursiveGaussian1D(image, sigma):
    (a,b,a_) = getCoefficients(sigma)
    (w,h) = image.shape
    #temp array for causal and anticausal values
    c = np.copy(image)
    ac = np.copy(image)
    c[c>=0]=0
    ac[ac>=0]=0

   # outImg = image.copy()
    gaussian = (1.0/(sigma * np.sqrt(2.0 * np.pi)))
    for x in range(4, w - 4):
        for y in range(4, h - 4):
            value=0.0
            for m in range(1, 5): 
                value += a[m-1]*image[x-m+1,y]
            for m in range(1, 5):
                value -= b[m-1]*c[x-m,y]
            c[x,y] = value

    for x in range(4, w - 4):
        for y in range(4, h - 4):
            value=0.0
            for m in range(1, 5): 
                value += a_[m-1]*image[x+m,y]
            for m in range(1, 5):
                value -= b[m-1]*ac[x+m,y]
            ac[x,y] = value

    return (c+ac)*gaussian
     
    
def main():  
    img = readIntensityImage('clock.jpg') #read the image as 2D array
    w,h = img.shape
    sigma = 10.0
    img = np.pad(img, 4,'constant')
    h = recursiveGaussian1D(img, sigma)
    out = recursiveGaussian1D(h.T, sigma)
    msc.toimage(out.T[4:w+4,4:w+4]).show()

if __name__ == '__main__':
    main()