import numpy
import cv2
import math

for k in range(0,1):
    name = "example/map"+str(k)+".txt"
    FILE = open(name)
    text = FILE.read().split()
    N = (int)(math.sqrt(len(text)))
    matrix = numpy.ones((N,N,3),numpy.uint8)*255

    resol = 2000
    str_resol = str(resol)+"x"+str(resol)
    seeds = str(10)

    count = 0
    for i in range(N):
        for j in range(N):
            color = int(text[count])
            matrix[i][j][0] = color%255
            matrix[i][j][1] = ((color+color%23 + color%53)%255)
            matrix[i][j][2] = ((color+color%31 + color%61)%255)
            #print(i*N + j)
            #print(color)
            if(i*N+j == color):
                #print(color)
                #print(j*N+i)
                #print(color)
                matrix[i][j][0]= 0
                matrix[i][j][1]= 0
                matrix[i][j][2]= 0
            count+=1

    filename = "images/"+str_resol+"/"+seeds+" seeds/image_" + str(N) + "_"+str(k)+".jpg"
    #print(filename)
    #cv2.imshow('image',matrix)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(filename, matrix) 
