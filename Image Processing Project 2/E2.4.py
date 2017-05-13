# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 16:51:32 2017

"""

import numpy as np
import scipy.misc as msc

def readIntensityImage(filename):
    f = msc.imread(filename, flatten=True).astype('float')
    return f
    
def showImage(a):
    msc.toimage(a).show()
 
def writeIntensityImage(f,filename):
    msc.toimage(f).save(filename+ '.jpg')

def getGaussianFilter(sigma):
    msize = int(np.ceil((2.575 * sigma) * 2 + 1))
    if (msize%2 == 0):
        msize+=1
    x = np.arange(msize)
    g1 = np.exp(-0.5 * ((x-msize/2) / sigma)**2)
    g1 /= g1.sum()
    g2 = np.outer(g1,g1)
    g2 /= g2.sum()
    return g2
    
    
def bilateral2d(image, sigma, rho):
    Gsigma = getGaussianFilter(sigma)
    (N,M) = image.shape
    (n,m) = Gsigma.shape
    image = np.pad(image, n/2,'constant')
    (N,M) = image.shape
    print m,n
    result = np.zeros((N,M))

    for y in range(n/2, N - n/2):
        for x in range(m/2,M - m/2):
            gamma = 0.0
            for j in range(-(n/2),n/2 ):
                for i in range( -(m/2), m/2 ):
                    Grho = np.exp(-0.5 * (abs(image[y+j,x+i]-image[y,x])/ rho) ** 2)
                    result[y,x] += image[y+j,x+i] * Gsigma[n/2-j,m/2-i] * Grho 
                    gamma += Gsigma[n/2-j,m/2-i] * Grho
            
            result[y,x] = result[y,x]/gamma          
                    
    return result[n/2: N-n/2, m/2: M-m/2]
    #return result
    
def main():
    inputImg = readIntensityImage('bauckhage.jpg')
    (w,h) = inputImg.shape
    sigma = 5.
    rho = 10.
    outImage = bilateral2d(inputImg, sigma, rho)
    showImage(outImage)
    #writeIntensityImage(outImage,'im4')
    
if __name__ == '__main__':
    main()