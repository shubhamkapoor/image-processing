import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

n = 512
x = np.linspace(0,2*np.pi, n)

# defining funtion and plotting the same
f = np.sin(x)
plt.plot(x, f, ’k-’)
plt.show()

# computing the Fourier Transform of sin(x) function and plotting the results
F = fft.fft(f)
w = fft.fftfreq(n)
plt.plot(w, F, ’k-’) #this plot gives warning as we try to plot a complex number; only absolute value is plotted
plt.show()

# here only the magnitude of complex number is plotted
plt.plot(w, np.abs(F), ’k-’)
plt.show()

# for better visualization, we scale down the fourier plot by using 'log' function
plt.plot(w, np.log(np.abs(F)), ’k-’)
plt.show()

# here fourier transformation is done on general form of the function
offset = 1.
amplitude = 2.
frequency = 16.
phase = np.pi

f = offset + amplitude * np.sin(frequency*x + phase)


F = fft.fft(f)
w = fft.fftfreq(n)

plt.plot(w,F,'k-')
plt.show()
