# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:09:39 2018

@author: cdm930PC_7
"""
import numpy as np
from numba import jit
import math
from time import time
import pickle
@jit(nopython=True)
def section(a,b,xi):
    L=15/100
    r=1.25/100
    sqrt_3=math.sqrt(3)
    m=np.zeros((4,4),np.float32)
    m[0,0]= 1/2 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 4 * (a + b) / r ** 10 * xi ** 10 / 3444525 + (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 3 * (a + b) / r ** 8 * xi ** 8 / 51030 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 2 * (a + b) / r ** 6 * xi ** 6 / 1215 + (2 * a - b) * (a ** 2 - a * b + b ** 2) * (a + b) / r ** 4 * xi ** 4 / 54 - (2 * a - b) * (a + b) / r ** 2 * xi ** 2 / 6
    m[0,1]=(2 * a - b) * (a ** 2 - a * b + b ** 2) ** 4 * (a - b) * sqrt_3 / r ** 10 * xi ** 10 / 3444525 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 3 * (a - b) * sqrt_3 / r ** 8 * xi ** 8 / 51030 + (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 2 * (a - b) * sqrt_3 / r ** 6 * xi ** 6 / 1215 - (2 * a - b) * (a ** 2 - a * b + b ** 2) * (a - b) * sqrt_3 / r ** 4 * xi ** 4 / 54 + (2 * a - b) * (a - b) * sqrt_3 / r ** 2 * xi ** 2 / 6 - sqrt_3/ 2
    m[0,2]= -2 / 688905 * (a ** 2 - a * b + b ** 2) ** 4 * (2 * a - b) * sqrt_3 / r ** 9 * xi ** 9 + 0.4e1 / 0.25515e5 * (a ** 2 - a * b + b ** 2) ** 3 * (2 * a - b) * sqrt_3/ r ** 7 * xi ** 7 - 0.2e1 / 0.405e3 * (a ** 2 - a * b + b ** 2) ** 2 * (2 * a - b) * sqrt_3 / r ** 5 * xi ** 5 + 0.2e1 / 0.27e2 * (a ** 2 - a * b + b ** 2) * (2 * a - b) * sqrt_3 / r ** 3 * xi ** 3 - (2 * a - b) * sqrt_3/ r * xi / 3
    m[0,3]=-(2 * a - b) * xi ** 2 / r * L * sqrt_3 / 6 + (2 * a - b) * (a ** 2 - a * b + b ** 2) * xi ** 4 / r ** 3 * L * sqrt_3 / 54 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / r ** 5 * L * sqrt_3 / 1215 + (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / r ** 7 * L * sqrt_3 / 51030 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 4 * xi ** 10 / r ** 9 * L * sqrt_3 / 3444525
    m[1,0]= -b * (a + b) * (a ** 2 - a * b + b ** 2) ** 4 * sqrt_3 / r ** 10 * xi ** 10 / 3444525 + b / r ** 8 * (a + b) * (a ** 2 - a * b + b ** 2) ** 3 * sqrt_3 * xi ** 8 / 51030 - b / r ** 6 * (a + b) * (a ** 2 - a * b + b ** 2) ** 2 * sqrt_3 * xi ** 6 / 1215 + sqrt_3 * b * (a ** 2 - a * b + b ** 2) * (a + b) / r ** 4 * xi ** 4 / 54 - b / r ** 2 * (a + b) * sqrt_3 * xi ** 2 / 6 + sqrt_3 / 2
    m[1,1]=1 / 2 + b * (a - b) * (a ** 2 - a * b + b ** 2) ** 4 / r ** 10 * xi ** 10 / 1148175 - b / r ** 8 * (a - b) * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / 17010 + b / r ** 6 * (a - b) * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / 405 - b / r ** 4 * (a - b) * (a ** 2 - a * b + b ** 2) * xi ** 4 / 18 + b / r ** 2 * (a - b) * xi ** 2 / 2
    m[1,2]=-0.2e1 / 0.229635e6 * (a ** 2 - a * b + b ** 2) ** 4 * b / r ** 9 * xi ** 9 + 0.4e1 / 0.8505e4 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 7 * b * xi ** 7 - 0.2e1 / 0.135e3 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 5 * b * xi ** 5 + 0.2e1 / 0.9e1 * (a ** 2 - a * b + b ** 2) / r ** 3 * b * xi ** 3 - b / r * xi
    m[1,3]=-L * b / r * xi ** 2 / 2 + b * (a ** 2 - a * b + b ** 2) * xi ** 4 / r ** 3 * L / 18 - b * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / r ** 5 * L / 405 + b * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / r ** 7 * L / 17010 - b * (a ** 2 - a * b + b ** 2) ** 4 * xi ** 10 / r ** 9 * L / 1148175
    m[2,0]=0.2e1 / 0.688905e6 * (a ** 2 - a * b + b ** 2) ** 4 * (a + b) * sqrt_3 / r ** 9 * xi ** 9 - 0.4e1 / 0.25515e5 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 7 * (a + b) * sqrt_3 * xi ** 7 + 0.2e1 / 0.405e3 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 5 * (a + b) * sqrt_3 * xi ** 5 - 0.2e1 / 0.27e2 * (a ** 2 - a * b + b ** 2) / r ** 3 * (a + b) * sqrt_3 * xi ** 3 + 1 / r * (a + b) * sqrt_3 * xi / 3
    m[2,1]=-0.2e1 / 0.229635e6 * (a ** 2 - a * b + b ** 2) ** 4 * (a - b) / r ** 9 * xi ** 9 + 0.4e1 / 0.8505e4 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 7 * (a - b) * xi ** 7 - 0.2e1 / 0.135e3 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 5 * (a - b) * xi ** 5 + 0.2e1 / 0.9e1 * (a ** 2 - a * b + b ** 2) / r ** 3 * (a - b) * xi ** 3 - 1 / r * (a - b) * xi
    m[2,2]=1 - 0.2e1 / 0.3e1 * xi ** 2 * (a ** 2 - a * b + b ** 2) / r ** 2 + 0.2e1 / 0.27e2 * xi ** 4 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 4 - 0.4e1 / 0.1215e4 * xi ** 6 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 6 + 0.2e1 / 0.25515e5 * xi ** 8 * (a ** 2 - a * b + b ** 2) ** 4 / r ** 8 - 0.4e1 / 0.3444525e7 * xi ** 10 * (a ** 2 - a * b + b ** 2) ** 5 / r ** 10
    m[2,3]= xi * L - 0.2e1 / 0.9e1 * xi ** 3 * (a ** 2 - a * b + b ** 2) / r ** 2 * L + 0.2e1 / 0.135e3 * xi ** 5 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 4 * L - 0.4e1 / 0.8505e4 * xi ** 7 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 6 * L + 0.2e1 / 0.229635e6 * xi ** 9 * (a ** 2 - a * b + b ** 2) ** 4 / r ** 8 * L
    m[3,3]=1
    return m



@jit(nopython=True)
def encode_point(params):
    p1=params[0]*(100**5)
    p2=params[1]*(100**4)
    p3=params[2]*(100**3)
    p4=params[3]*(100**2)
    p5=params[4]*(100**1)
    p6=params[5]
    return p1+p2+p3+p4+p5+p6

    
@jit(nopython=True)
def generate_points(intervals):
    params=np.empty((325660672,9),np.float32)    
    index =0
    count =0
    for r1 in intervals:
        for r2 in intervals:
            for r3 in intervals:
                params[index][0:2]=r1*100
                params[index][2:4]=r2*100
                params[index][4:6]=r3*100
                m1=section(r1[0],r1[1],1)
                m2=section(r2[0],r2[1],1)
                m3=section(r3[0],r3[1],1)
                result= m1.dot(m2)
                result=result.dot(m3)
                coord=result.T[3][0:3]
                params[index][6:9]=coord*100
                index +=1
                count +=1
                if(count == 100000):
                    print(index/325660672)
                    count =0
    return params



def encode_cube(cube):
    return (cube[0]+41)*10000+(cube[1]+41)*100+(cube[2]+30)

def decode_cube(cube):
    x=int(cube//10000)-41;
    y=int(cube%10000//100)-41
    z=int(cube%100)-30
    return np.array([x,y,z]) 





#input points
"""
import os
os.chdir('C:\\Users\\cdm930PC_7\Desktop\\masterthesis\\3-sec')
file=open('list.txt')
intervals=[]
for line in file:
    line=line.split(',')
    intervals.append(np.array((float(line[0]),float(line[1])),np.float32))
file.close()   
"""

#generateparams
"""
params=generate_points(intervals)                                     
np.save("points",params)
cubes=params[:,6:9]
cubes=np.floor(cubes/1)
np.save("cubes",cubes)
unique_rows = np.unique(cubes, axis=0)
np.save("unique_rows.npy",unique_rows)
"""
   
params=np.load("points.npy")
cubes=np.load("cubes.npy")
unique_rows=np.load("unique_rows.npy")


all_encode=(cubes[:,0]+41)*10000+(cubes[:,1]+41)*100+(cubes[:,2]+30)
sorted_indexs=np.argsort(all_encode)
cube_list=all_encode[sorted_indexs]#sorted encoded cubes
sorted_params=params[sorted_indexs]#sorted points

cube_set=set(list(cube_list))#set version of cube_list
with open('cube_set.pickle','wb') as handle:
    pickle.dump(cube_set,handle)
#np.save('cube_list',cube_list) c
cube_list=np.load('cube_list.npy')
#np.save('sorted_params',sorted_params)
sorted_params=np.load("sorted_params.npy")
index_lookup={}
start=0
end=0
while(end<=len(cube_list)):
    if(end<len(cube_list) and cube_list[end]==cube_list[start] ):
        end +=1
    else:
        index_lookup[int(cube_list[start])]=(start,end)
        start=end
        end +=1
##index lookup contains the ranges of the indexes of cubes in the list. since they are not unique, every instance of the cube corresponds to a point/param.
        ##ergo the range listed here is showing the range for params for a given cube
    
        
with open('index_lookup.pickle','wb') as handle:
    pickle.dump(index_lookup,handle)

with open('index_lookup.pickle','rb') as handle:
    index_lookup=pickle.load(handle)

import pickle
from igraph import *

l=list(cube_set)#list of set of sorted cubes
l=sorted(l)
cube_lookup={}
for i in range(len(l)):
    cube_lookup[int(l[i])]=i
with open('cube_lookup.pickle','wb') as handle:
    pickle.dump(cube_lookup,handle)
count=0
counter=1
edges=[]
weights=[]
for cube_encoding in l:
    num_cube=decode_cube(cube_encoding)
    for a in range(num_cube[0]-1,num_cube[0]+2):
        for b in range(num_cube[1]-1,num_cube[1]+2):
            for c in range(num_cube[2]-1,num_cube[2]+2):
                cube=encode_cube([a,b,c])
                if(cube in cube_set and cube != cube_encoding):
                    edges.append((cube_lookup[cube_encoding],cube_lookup[cube]))
                    p1=num_cube
                    p2=np.array([a,b,c])
                    
                    dist=np.linalg.norm(p1-p2)
                    weights.append(dist)
    count+=1
    if(count==100000):
        print(counter*100000)
        count =0
        counter +=1
g=Graph()
g.add_vertices(len(cube_list))
g.add_edges(edges)
g.es['weight']=weights
with open('cube_graph.pickle','wb') as handle:
    pickle.dump(g,handle)
with open('cube_graph.pickle','rb') as handler:
    graph=pickle.load(handler)
with open('cube_lookup.pickle','rb') as handle:
    lookup=pickle.load(handle)

#
#params=np.empty((325660672,9),np.float32)    
#index =0
#count =0
#for r1 in intervals:
#    for r2 in intervals:
#        for r3 in intervals:
#            params[index][0:2]=r1*100
#            params[index][2:4]=r2*100
#            params[index][4:6]=r3*100
#            m1=section(r1[0],r1[1],1)
#            m2=section(r2[0],r2[1],1)
#            m3=section(r3[0],r3[1],1)
#            result= m1.dot(m2)
#            result=result.dot(m3)
#            coord=result.T[3][0:3]
#            params[index][6:9]=coord*100
#            index +=1
#            count +=1
#            if(count == 100000):
#                print(index/325660672)
#                count =0            