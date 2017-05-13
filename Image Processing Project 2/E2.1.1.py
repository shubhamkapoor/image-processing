import numpy as np
import scipy as sp
import scipy.misc as msc

# function used to read intensity images.
def readIntensityImage(filename):    
    f = msc.imread(filename, flatten=True).astype('float')
    return f

# function to generate and save intensity images.    
def writeIntensityImage(f,filename):
    msc.toimage(f, cmin = 0, cmax = 255).save(filename+'x'+filename + '.jpg')

# Returns a normalized 2D gauss kernel array for convolutions
def createGaussianKernel(size, sigma):
    y, x = sp.mgrid[-size:size+1, -size:size+1]
    kernel = sp.exp(-(x**2 / float(2*sigma*sigma)+y**2/float(2*sigma*sigma)))
    return kernel / kernel.sum()
    print kernel / kernel.sum()
    
# This function displays the intensity image in new figure window.
def showImage(a):
    msc.toimage(a,cmin=0,cmax=255).show()

# naive 2d convolution
def convolve2d(image, kernel):
    (N,M) = image.shape
    (n,m) = kernel.shape
    image = np.pad(image, n/2,'constant')
    (N,M) = image.shape
    result = np.zeros((N,M))
    for y in range(n/2, N - n/2):
        for x in range(m/2,M - m/2):
            for j in range(-(n/2),n/2 ):
                for i in range( -(m/2), m/2 ):
                    result[y,x] += image[y+j,x+i] * kernel[n/2-j,m/2-i]  
    
    return result[n/2: N-n/2, m/2: M-m/2]
    
def main():
    inputImg = readIntensityImage('bauckhage.jpg') #read the image as 2D array
    
    size = input("enter size of kernel (for example 3, 5, etc)")
    sigma = (size - 1.0) / (2.0 * 2.575)
    kernel = createGaussianKernel(int(size/2), sigma) 
    
    afterConvolution = convolve2d(inputImg, kernel)
    showImage(afterConvolution)
    #writeIntensityImage(afterConvolution, str(size))
    
if __name__ == '__main__': # boilerplate code to invoke main()
    main()


    

