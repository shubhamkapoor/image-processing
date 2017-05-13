import numpy as np
import scipy as sp
import scipy.misc as msc
import scipy.ndimage as img
import matplotlib.pyplot as plt
import time
import numpy.fft as fft

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
    
    
# This function displays the intensity image in new figure window.
def showImage(a):
    msc.toimage(a,cmin=0,cmax=255).show()

# naive 2d convolution
def convolve2d(image, kernel):
    (N,M) = image.shape
    (n,m) = kernel.shape
    result = np.zeros((N,M))
    for y in range(n/2, N - n/2):
        for x in range(m/2,M - m/2):
            for j in range(-(n/2),n/2 ):
                for i in range( -(m/2), m/2 ):
                    result[y,x] += image[y+j,x+i] * kernel[n/2-j,m/2-i]  
    return result

# Returns a normalized 1D gaussian kernel array 
def Gaussian1D(size, sigma):
    x =np.arange(size)
    kernel = np.exp(-0.5 * ((x-size/2) / sigma)**2)
    return kernel / kernel.sum()

def padKernel(kernel,w,h):
    m,n = kernel.shape
    if m%2==0:
        kernalPad = np.pad(kernel, ((w-n)/2, 'constant'))
    else:
        kernelPad = np.pad(kernel, ((w-n)/2, (w-n)/2 +1), 'constant')
      
    return kernelPad

    
def main():
    inputImg = readIntensityImage('bauckhage.jpg') #read the image as 2D array
    (w,h) = inputImg.shape # return the width and height of the image
    sizeList = []
    runTimeList = []
    for i in np.arange(3,22,2):
        startTime = time.time()
        size = i
        sigma = (size - 1.0) / (2.0 * 2.575)
        kernel = createGaussianKernel(int(size/2), sigma) 
        afterConvolution = convolve2d(inputImg, kernel)
        writeIntensityImage(afterConvolution, str(size))
        runTime = time.time() - startTime
        sizeList.append(size)
        runTimeList.append(runTime)

    plt.subplot(1,3,1)
    plt.plot(sizeList, runTimeList)
    plt.ylabel('running time in seconds')
    plt.xlabel('size of mask')

    sizeList = []
    runTimeList = []
    for i in np.arange(3,22,2):
        startTime = time.time()
        size = i
        sigma = (size - 1.0) / (2.0 * 2.575)
        kernel = Gaussian1D(size, sigma)
        convX = img.convolve1d(inputImg, kernel, mode='constant', cval=0.0)
        convY = img.convolve1d(convX.T, kernel, mode='constant', cval=0.0)
        outputImg = convY.T
        writeIntensityImage(outputImg, str(size))
        runTime = time.time() - startTime
        sizeList.append(size)
        runTimeList.append(runTime)


    plt.subplot(1,3,2)
    plt.plot(sizeList, runTimeList)
    plt.ylabel('running time in seconds')
    plt.xlabel('size of mask')

    sizeList = []
    runTimeList = []
    for i in np.arange(3,22,2):
        startTime = time.time()
        size = i
        sigma = (size - 1.0) / (2.0 * 2.575)
        kernel = createGaussianKernel(size, sigma)
        kernalPad = padKernel(kernel,w,h)
        kernalPad_shift = fft.fftshift(kernalPad)
        ftimage = fft.fft2(inputImg)
        ftkernel = fft.fft2(kernalPad_shift)
        finalImage = np.abs(np.fft.ifft2(np.multiply(ftimage, ftkernel)))
        writeIntensityImage(finalImage, str(size))
        runTime = time.time() - startTime
        sizeList.append(size)
        runTimeList.append(runTime)


    plt.subplot(1,3,3)
    plt.plot(sizeList, runTimeList)
    plt.ylabel('running time in seconds')
    plt.xlabel('size of mask')

    plt.show()    

    
if __name__ == '__main__': # boilerplate code to invoke main()
    main()


    

