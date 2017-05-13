import numpy as np
import scipy.misc as msc
import scipy.ndimage as img

# function used to read intensity images.
def readIntensityImage(filename):    
    f = msc.imread(filename, flatten=True).astype('float')
    return f

# function to generate and save intensity images.    
def writeIntensityImage(f,filename):
    msc.toimage(f, cmin = 0, cmax = 255).save(filename+'x'+filename + '1D.jpg')

# Returns a normalized 1D gaussian kernel array 
def Gaussian1D(size, sigma):
    x =np.arange(size)
    kernel = np.exp(-0.5 * ((x-size/2) / sigma)**2)
    return kernel / kernel.sum()


def main():
    inputImg = readIntensityImage('bauckhage.jpg') #read the image as 2D array
    
    size = input("enter size of kernel (for example 3, 5, etc): ")
    sigma = (size - 1.0) / (2.0 * 2.575)
    kernel = Gaussian1D(size, sigma)
    
    convX = img.convolve1d(inputImg, kernel, mode='constant', cval=0.0)
    convY = img.convolve1d(convX.T, kernel, mode='constant', cval=0.0)
    outputImg = convY.T
    
    msc.toimage(outputImg,cmin=0,cmax=255).show()
    #writeIntensityImage(outputImg, str(size))
    
if __name__ == '__main__': # boilerplate code to invoke main()
    main()


    

