
# coding: utf-8

# In[45]:

from PIL import Image
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
import glob

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

for file in glob.glob("DSC*.jpg"):
    imageFiles = np.append(imageFiles, file)
    currImage = Image.open(file)
    print(file)
#slika = Image.open("DSC07592.JPG")
    arej = np.asarray(currImage.convert('L'))
    format = currImage.size
    #slika.show()
    #print("Dimenzije slike su:", format)
    
    niz = np.asarray([])
    x = np.asarray([])
    yprvo = -1
    
    for y in range (0, format[1]):
        intenzitet = arej[y]
        prvi = -1
        poslednji = -1
        for i in range(format[0]):
            if intenzitet[i] > 30 :
                if prvi == -1 :
                    prvi = i
                else:
                    poslednji = i
    
        if prvi ==-1:
            pass
        if poslednji == -1 :
            pass
        rastojanje = poslednji - prvi
        if rastojanje > 30 and rastojanje < 2000:
            if yprvo == -1:
                yprvo = y
            rastojanje = poslednji - prvi
            #niz = np.append(niz,rastojanje)
            x = np.append(x, y)
            x = np.append(x, y)
            niz = np.append(niz, prvi)
            niz = np.append(niz, poslednji)        
            #x.append(y-yprvo)
    
    a = ellip_fit(x[1:100], niz[1:100])
    axes = ellip_ose(a)
    m, v = axes                  #m = mala poluosa; v = velika poluosa
    #print('ove su poluose', m, v)
    #print('Velika polusa: ', max(m,v))
    outputVals = np.append(outputVals, max(m,v))
    currImage.close()
    #plt.scatter(x, niz)
    #plt.show()
print('Gotovo!')
plt.plot(outputVals)
plt.show()


