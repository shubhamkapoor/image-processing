import numpy as np
import scipy.misc as msc


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
    dist = np.sqrt(np.square(X - w/2)  + np.square(Y - h/2))
    return dist
  
def maskMat(rMin, rMax, w, h):
    # This function returns the logical matrix for elements whose distance
    # from center lies between rMin and rMax
    
    dist = distanceMat(w, h)
    mask = ((dist>=rMin) & (dist<=rMax))
    return mask
    
def showImage(a):
    # This function displays the intensity image in new figure window.
    
    msc.toimage(a,cmin=0,cmax=255).show()
    
def main():
    inputImg = readIntensityImage('clock.jpg') #read the image as 2D array
    (w,h) = inputImg.shape # return the width and height of the image
    rMin,rMax = (50,65) # assign the rMin and rMax values 
    mask = maskMat(rMin,rMax,w,h) #returns the mask 2D array for elements that satisfy the given conditions
    inputImg[mask] = 0 # supress the intensities of all elements pointed by the 'mask' 2D array
    showImage(inputImg)
    
if __name__ == '__main__': # boilerplate code to invoke main()
    main()


    

