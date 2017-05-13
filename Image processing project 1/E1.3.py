import numpy as np
import scipy.misc as msc
import numpy.fft as fft
import matplotlib.pyplot as plt

def readIntensityImage(filename):
    # function used to read intensity images.
    
    f = msc.imread(filename, flatten=True).astype('float')
    return f
    
def writeIntensityImage(f,filename):
    # function to generate and save intensity images.
    
    msc.toimage(f, cmin=0, cmax=255).save(filename)
    
def distanceMat(w, h):
    # This function calculates the distance of each pixmap element from
    # the center of ix map and return a the distance matrix.
    
    xs = range(w)
    X,Y = np.meshgrid(xs,xs)
    dist = np.sqrt(np.square(X - w/2) + np.square(Y - h/2))
    return dist    
    
def maskMat(rMin, rMax, w, h):
    # This function returns the logical matrix for elements whose distance
    # from center lies between rMin and rMax
    
    dist = distanceMat(w, h)
    mask = ((dist>=rMin) & (dist<=rMax))
    return mask

def showSpectrum(Fs):
    # generates the visualization of Fourier transform spectrum
    
    return plt.imshow(np.log(np.abs(Fs)), cmap='gray') 


def main():
    inputImg = readIntensityImage('clock.jpg') #read image as 2D array
    (w,h) = inputImg.shape # return width and height of image
    rMin,rMax = (20,45) # assign min and max distance values
    mask = maskMat(rMin,rMax,w,h) # return the logical 2D array for elements the satisfy conditions
    
    F = fft.fft2(inputImg) # Calculate the Fast Fourier transformation of image
    Fs = fft.fftshift(F) # brings the zero frequency to center
    showSpectrum(Fs) 
    
    Fs[~mask] = 0 # supress the intensities of all elements out of the band 
    plt.imshow(np.abs(Fs), cmap='gray') 
    
    Fsi = fft.ifftshift(Fs) #inverse shift on fouries transform
    Fi = fft.ifft2(Fsi) # inverse fourier transform to compute the function i.e. image
    
    plt.imshow(abs(Fi), cmap='gray')
    
    
if __name__ == '__main__':
    main()
