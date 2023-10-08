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
