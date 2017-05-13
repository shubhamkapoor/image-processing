import numpy as np
import scipy.misc as msc
import numpy.fft as fft
import matplotlib.pyplot as plt

def readIntensityImage(filename):
    # function used to read intensity images.
    
    f = msc.imread(filename, flatten=True).astype('float')
    return f
    
def showImage(a):
    # This function displays the intensity image in new figure window.
    
    msc.toimage(a,cmin=0,cmax=255).show()
    
    
def main():
    G = readIntensityImage('bauckhage.jpg') # read first image as 2D array
    H = readIntensityImage('clock.jpg') # read second image as 2D array
    
    Fg = fft.fftshift(fft.fft2(G)) # Calculate the Fourier transform of first image
    Fh = fft.fftshift(fft.fft2(H)) # Calculate the Fourier transform of second image
    
    magnitude_F = np.abs(Fg) # extract the magnitudes of all frequency vectors from first image
    phase_G = np.angle(Fh) # extract the phase of all frequency vectors from second image
    
    Fk = np.multiply(magnitude_F, np.exp(1j * phase_G)) # Generate frequency vectors using magnitude of image 1 and phase from image 2
    K = fft.ifft2(fft.ifftshift(Fk)) # calculate the inverse fourier transform 'Fk' of new function 'k' 
    
    showImage(np.abs(K))
    
    
if __name__ == '__main__':
    main()



