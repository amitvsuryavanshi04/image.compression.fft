 from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize']=[5,5]
plt.rcParams.update({'font.size':18})

Img=imread(os.path.join('test.jpg')) #this command reads the image
ImgGray=np.mean(Img,-1) #this line converts the RGB image to grayscale

plt.figure()
plt.imshow(ImgGray,cmap='gray')#to get output of the grayscale image
plt.axis('off') #as we don't need axis so it's kept off

#so now we will start the compression part using fft2 which is a built in function in the numpy lib
ImgGrayfft=np.fft.fft2(ImgGray)
ImgGrayfft_sort=np.sort(np.abs(ImgGrayfft.reshape(-1)))

#we need zero out all the unnecessary fft coefficiencts and retain the required or imp coefficients for compression
for keep in(0.1,0.05,0.001,0.0002):
  threshold=ImgGrayfft_sort[int(np.floor((1-keep)*len(ImgGrayfft_sort)))]
  ind=np.abs(ImgGrayfft)>threshold
  ImgGrayfft_low=ImgGrayfft*ind
  Imglow=np.fft.ifft2(ImgGrayfft_low).real
  plt.figure()
  plt.imshow(Imglow,cmap='gray')
  plt.axis('off')
  plt.title('compressed image:retained:'+str(keep*100)+'%')
plt.show()

#as you all see below the 10% retained image is much clearer and it has a smaller storage size
#hence we can conclude that fft can perform the image compression very easily
