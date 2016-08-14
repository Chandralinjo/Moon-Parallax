
# coding: utf-8

# In[45]:

from PIL import Image
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
import glob

#Ellipse fitting function, returns ellipse parameters
#Only method for calculating semiaxis is used, the rest is available in
#the Utilities repository
def ellip_fit(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a
    
#Determining semiaxis from ellipse parameters
def ellip_ose( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

outputVals = np.empty([0,1], dtype=np.float)
imageFiles = np.empty([0,1], dtype=np.str)

#TODO: Enable setting input directory

for file in glob.glob("DSC*.jpg"):
    imageFiles = np.append(imageFiles, file)
    currImage = Image.open(file)
    print(file)

    arej = np.asarray(currImage.convert('L'))
    format = currImage.size
    
    niz = np.asarray([])
    x = np.asarray([])
    yprvo = -1
    
    #Passing through image lines and finding edges of the Moon
    for y in range (0, format[1]):
        intenzitet = arej[y]
        first = -1
        last = -1
        for i in range(format[0]):
            if intenzitet[i] > 30 :
                if first == -1 :
                    first = i
                else:
                    last = i
    
        if first ==-1:
            pass
        if last == -1 :
            pass
        rastojanje = last - first
        if rastojanje > 30 and rastojanje < 2000:
            if yprvo == -1:
                yprvo = y
            rastojanje = last - first
            #niz = np.append(niz,rastojanje)
            x = np.append(x, y)
            x = np.append(x, y)
            niz = np.append(niz, first)
            niz = np.append(niz, last)        
            #x.append(y-yprvo)
    
    #Fitting the elipse and depositing current image major semiaxis to an array
    a = ellip_fit(x, niz)
    axes = ellip_ose(a)
    m, v = axes                 
    outputVals = np.append(outputVals, max(m,v))
    currImage.close()

print('Finished!')
#TODO: Export array of semiaxis to a file
#plt.plot(outputVals)
#plt.show()


