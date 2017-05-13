import numpy as np
import scipy.misc as msc
import scipy.ndimage as img

# function used to read intensity images.
def readIntensityImage(filename):    
    f = msc.imread(filename, flatten=True).astype('float')
    return f

# function to generate and save intensity images.    
def writeIntensityImage(f,filename):
    msc.toimage(f, cmin = 0, cmax = 255).save(filename+'x'+filename + '.jpg')

# Returns a normalized 2D gauss kernel array for convolutions
def createGaussianKernel(size, sigma):
    x =np.arange(size)
    g1D = np.exp(-0.5 * ((x-size/2) / sigma)**2)
    g1D = g1D / g1D.sum()
    kernel = np.outer(g1D, g1D)
    return kernel / kernel.sum()

# applying gaussian filter    
def applyGaussianFilter(image):
	return img.gaussian_filter(image, sigma = 3)
    
# This function displays the intensity image in new figure window.
def showImage(a):
    msc.toimage(a,cmin=0,cmax=255).show()

    
def main():
    
    inputImg2 = readIntensityImage('clock.jpg') #read the image as 2D array
    
    size = input("enter size of kernel (for example 3, 5, etc): ")
    sigma = (size - 1.0) / (2.0 * 2.575)
    kernel = createGaussianKernel(size, sigma)
    dx, dy = np.gradient(kernel) 
    convolutionGradDx = img.convolve(inputImg2, dx, mode='constant', cval=0.0)
    convolutionGradDy = img.convolve(inputImg2, dy, mode='constant', cval=0.0)
    
    gradImg = np.sqrt(np.square(convolutionGradDx) + np.square(convolutionGradDy))
    
    showImage(gradImg)
    #writeIntensityImage(gradImg, str(size) + 'clock')
    
if __name__ == '__main__': # boilerplate code to invoke main()
    main()


    

