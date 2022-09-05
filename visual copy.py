import numpy
import cv2
import math
import random as rand

rand.seed(1)
colors = []
for i in range(4000):
    cr = rand.randint(0,255)
    cg = rand.randint(0,255)
    cb = rand.randint(0,255)
    colors.append([cr,cg,cb])
for k in range(-1,1):
    name = "example/map"+str(k)+".txt"
    FILE = open(name)
    text = FILE.read().split()
    N = (int)(math.sqrt(len(text)))
    matrix = numpy.ones((N,N,3),numpy.uint8)*255

    resol = 2000
    str_resol = str(resol)+"x"+str(resol)
    seeds = str(500)
    count = 0
    for i in range(N):
        for j in range(N):
            color = int(text[count])
            matrix[i][j][0] = colors[color][0]
            matrix[i][j][1] = colors[color][1]
            matrix[i][j][2] = colors[color][2]
            #print(i*N + j)
            #print(color)
            if(i*N+j == color):
                #print(color)
                #print(j*N+i)
                #print(color)
                matrix[i][j][0]= 255
                matrix[i][j][1]= 255
                matrix[i][j][2]= 255
            count+=1

    filename = "images_ref/"+str_resol+"/"+seeds+" seeds/image_" + str(N) + "_"+str(k)+".jpg"
    #print(filename)
    #cv2.imshow('image',matrix)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(filename, matrix) 
