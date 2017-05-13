import numpy as np
import scipy.misc as msc
import numpy.fft as fft
import scipy.ndimage as img

# function used to read intensity images.
def readIntensityImage(filename):    
    f = msc.imread(filename, flatten=True).astype('float')
    return f

# function to generate and save intensity images.    
def writeIntensityImage(f,filename):
    msc.toimage(f, cmin = 0, cmax = 255).save(filename+'x'+filename + 'fourier.jpg')

# Returns a normalized 2D gauss kernel array for convolutions
def createGaussianKernel(size, sigma):
    x =np.arange(size)
    g1D = np.exp(-0.5 * ((x-size/2) / sigma)**2)
    g1D = g1D / g1D.sum()
    kernel = np.outer(g1D, g1D)
    return kernel / kernel.sum()
    
# This function displays the intensity image in new figure window.
def showImage(a):
    msc.toimage(a,cmin=0,cmax=255).show()
    

def padKernel(kernel,w,h):
    m,n = kernel.shape
    if m%2==0:
        kernalPad = np.pad(kernel, ((w-n)/2, 'constant'))
    else:
        kernelPad = np.pad(kernel, ((w-n)/2, (w-n)/2 +1), 'constant')
        
    return kernelPad


def main():
    inputImg = readIntensityImage('bauckhage.jpg') #read the image as 2D array
    w,h = inputImg.shape
    
    size = input("enter size of kernel (for example 3, 5, etc)")
    sigma = (size - 1.0) / (2.0 * 2.575)
    kernel = createGaussianKernel(size, sigma)
    
    kernalPad = padKernel(kernel,w,h)
    kernalPad_shift = fft.fftshift(kernalPad)
    
    ftimage = fft.fft2(inputImg)
    ftkernel = fft.fft2(kernalPad_shift)
    finalImage = np.abs(np.fft.ifft2(np.multiply(ftimage, ftkernel)))
    showImage(finalImage)
    #writeIntensityImage(finalImage, str(size))
    
if __name__ == '__main__': # boilerplate code to invoke main()
    main()


    

