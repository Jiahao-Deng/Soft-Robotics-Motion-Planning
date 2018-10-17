# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 18:34:30 2018

@author: cdm930PC_7
"""

'''
Data we need to import into the program:
1. index_lookup (look up the indexs of params given cube encodings)
2. sorted params
3. cube_graph
4. cube_lookup

Functions we need:
1. distance function
2. cube_encoding
3. cube_decoding
4. section function

'''
import numpy as np
import pickle
from igraph import Graph
import os
import math
from numba import jit
from time import time
os.chdir('C:\\Users\\cdm930PC_7\Desktop\\masterthesis\\3-sec')



#def distance(m1,m2):
#    L=0.15
#    r=0.0125
#    sqrt_3=math.sqrt(3)
#    # calculate the parameters of m1
#    a=m1[:,0].reshape((-1,1))
#    b=m1[:,1].reshape((-1,1))
#    m1_phi1=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)
#    m1_lambda1=(1/2)*L*sqrt_3*r/np.sqrt(a**2-a*b+b**2)
#    m1_theta1=np.arctan2(b*sqrt_3, 2*a-b)
#    
#    a=m1[:,2].reshape((-1,1))
#    b=m1[:,3].reshape((-1,1))
#    m1_phi2=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)
#    m1_lambda2=(1/2)*L*sqrt_3*r/np.sqrt(a**2-a*b+b**2)
#    m1_theta2=np.arctan2(b*sqrt_3, 2*a-b)
#    
#    a=m1[:,4].reshape((-1,1))
#    b=m1[:,5].reshape((-1,1))
#    m1_phi3=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)
#    m1_lambda3=(1/2)*L*sqrt_3*r/np.sqrt(a**2-a*b+b**2)
#    m1_theta3=np.arctan2(b*sqrt_3, 2*a-b)
#    
#    m1=np.concatenate((m1_phi1,m1_lambda1,m1_theta1,m1_phi2,m1_lambda2,m1_theta2,m1_phi3,m1_lambda3,m1_theta3), axis=1)
#    
#    # calculate the parameters of m2
#    a=m2[:,0].reshape((-1,1))
#    b=m2[:,1].reshape((-1,1))
#    m2_phi1=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)
#    m2_lambda1=(1/2)*L*sqrt_3*r/np.sqrt(a**2-a*b+b**2)
#    m2_theta1=np.arctan2(b*sqrt_3, 2*a-b)
#    
#    a=m2[:,2].reshape((-1,1))
#    b=m2[:,3].reshape((-1,1))
#    m2_phi2=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)
#    m2_lambda2=(1/2)*L*sqrt_3*r/np.sqrt(a**2-a*b+b**2)
#    m2_theta2=np.arctan2(b*sqrt_3, 2*a-b)
#    
#    a=m2[:,4].reshape((-1,1))
#    b=m2[:,5].reshape((-1,1))
#    m2_phi3=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)
#    m2_lambda3=(1/2)*L*sqrt_3*r/np.sqrt(a**2-a*b+b**2)
#    m2_theta3=np.arctan2(b*sqrt_3, 2*a-b)
#    
#    m2=np.concatenate((m2_phi1,m2_lambda1,m2_theta1,m2_phi2,m2_lambda2,m2_theta2,m2_phi3,m2_lambda3,m2_theta3), axis=1)
#    
#    # expand the parameters to a 3-dimentional arrays to vectorize
#    shape1=m1.shape
#    shape2=m2.shape
#    m1=np.repeat(m1[:, :, np.newaxis], shape2[0], axis=2)
#    m2 =np.repeat(m2[:,:, np.newaxis], shape1[0], axis=2).T
#    distances=np.linalg.norm(m1-m2,axis=1).T
#    return np.argmin(distances,axis=1),np.min(distances,axis=1)

def distance(m1,m2):
    L=0.15
    r=0.0125
    sqrt_3=math.sqrt(3)
    # calculate the parameters of m1
    a=m1[:,0].reshape((-1,1))
    b=m1[:,1].reshape((-1,1))
    m1_phi1=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)*1.1
    
    m1_theta1=np.arctan2(b*sqrt_3, 2*a-b)*1.1
    
    a=m1[:,2].reshape((-1,1))
    b=m1[:,3].reshape((-1,1))
    m1_phi2=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)
    m1_theta2=np.arctan2(b*sqrt_3, 2*a-b)
    
    a=m1[:,4].reshape((-1,1))
    b=m1[:,5].reshape((-1,1))
    m1_phi3=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)
    
    m1_theta3=np.arctan2(b*sqrt_3, 2*a-b)
    
    m1=np.concatenate((m1_phi1,m1_theta1,m1_phi2,m1_theta2,m1_phi3,m1_theta3), axis=1)
    
    # calculate the parameters of m2
    a=m2[:,0].reshape((-1,1))
    b=m2[:,1].reshape((-1,1))
    m2_phi1=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)*1.1
   
    m2_theta1=np.arctan2(b*sqrt_3, 2*a-b)*1.1
    
    a=m2[:,2].reshape((-1,1))
    b=m2[:,3].reshape((-1,1))
    m2_phi2=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)
    
    m2_theta2=np.arctan2(b*sqrt_3, 2*a-b)
    
    a=m2[:,4].reshape((-1,1))
    b=m2[:,5].reshape((-1,1))
    m2_phi3=2/3*(sqrt_3*np.sqrt(a**2-a*b+b**2)/r)
    
    m2_theta3=np.arctan2(b*sqrt_3, 2*a-b)
    
    m2=np.concatenate((m2_phi1,m2_theta1,m2_phi2,m2_theta2,m2_phi3,m2_theta3), axis=1)
    
    # expand the parameters to a 3-dimentional arrays to vectorize
    shape1=m1.shape
    shape2=m2.shape
    m1=np.repeat(m1[:, :, np.newaxis], shape2[0], axis=2)
    m2 =np.repeat(m2[:,:, np.newaxis], shape1[0], axis=2).T
    distances=np.linalg.norm(m1-m2,axis=1).T
    return np.argmin(distances,axis=1),np.min(distances,axis=1)  

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
def robot(p,xi_list=[1.0,1.0,1.0]):
    m1=section(p[0],p[1],xi_list[0])
    m2=section(p[2],p[3],xi_list[1])
    m3=section(p[4],p[5],xi_list[2])
    m=m1.dot(m2)
    m=m.dot(m3)
    coord=m.T[3][0:3]
    return coord
def encode_cube(cube):
    return (cube[0]+41)*10000+(cube[1]+41)*100+(cube[2]+30)

def decode_cube(cube):
    x=int(cube//10000)-41;
    y=int(cube%10000//100)-41
    z=int(cube%100)-30
    return np.array([x,y,z]) 
  

#
#l=[]
#for key in index_lookup:
#    start,end=index_lookup[key]
#    l.append(end-start)

#path=self.graph.get_shortest_paths(13333,to=end,weights='weight')
class Planner:
    def __init__(self):
        with open('cube_graph.pickle','rb') as handler:
            self.cube_graph=pickle.load(handler)
        self.params=np.load('sorted_params.npy')
        with open('cube_lookup.pickle','rb') as handle:
            self.cube_lookup=pickle.load(handle)
        with open('index_lookup.pickle','rb') as handle:
            self.index_lookup=pickle.load(handle)
        self.no_duplicate_cubes=list(self.cube_lookup.keys())
        self.sorted_points=np.load("sorted_params.npy")
        self.sorted_points=self.sorted_points[:,0:6]
            
    def encoding_to_index(self,encoding):
        return self.cube_lookup[encoding]
    def index_to_encoding(self,index):
        return self.no_duplicate_cubes[index]
    def encode_cube(self,cube):
        return (cube[0]+41)*10000+(cube[1]+41)*100+(cube[2]+30)
    def decode_cube(self,cube):
        x=int(cube//10000)-41;
        y=int(cube%10000//100)-41
        z=int(cube%100)-30
        return np.array([x,y,z]) 
    def reload_graph(self):
        with open('cube_graph.pickle','rb') as handler:
                self.cube_graph=pickle.load(handler)
    def reload_other(self):
        self.params=np.load('sorted_params.npy')
        with open('cube_lookup.pickle','rb') as handle:
            self.cube_lookup=pickle.load(handle)
        with open('index_lookup.pickle','rb') as handle:
            self.index_lookup=pickle.load(handle)
        with open("cube_set.pickle", "rb") as handle:
            self.no_duplicate_cubes=list(pickle.load(handle))
            sorted(self.no_duplicate_cubes)
            
    def distance_helper(self,cube_points,start):
        path_len=len(cube_points)
        if path_len==1:
            print("At Destination")
        distance_info=[]
        #first loop
        #this is the first item in the list
        prev=start/100#starting point
        next = cube_points[0]
        dis=distance((prev).reshape(1,6),next/100)
        distance_info.append(dis)
        #rest of the loops
        for i in range(1, path_len):
            prev=cube_points[i-1]
            next=cube_points[i]
            min_indexs,min_dis=distance(prev/100,next/100)
            min_dis += np.take(distance_info[i-1][1],min_indexs)
            distance_info.append((min_indexs,min_dis))
        return distance_info
    
    def find_cube_path(self, s, f):
        s_encoded=self.encode_cube(s)
        f_encoded=self.encode_cube(f)
        start=self.encoding_to_index(s_encoded)
        finish=self.encoding_to_index(f_encoded)
        
        path=self.cube_graph.get_shortest_paths(start,to=finish,weights='weight')
        result=[]
        for p in path[0]:
            a=self.index_to_encoding(p)
            num_cube=np.array(self.decode_cube(a))
            result.append(num_cube)
        return result
    def basic_path(self, start_config, end_point,filter_params=None,obs=None,delta=1):
        start_point=robot(start_config/100,[1.0,1.0,1.0])*100
        start_cube=np.floor(start_point/delta)
        end_cube=np.floor(end_point/delta)
        cube_path=self.find_cube_path(start_cube,end_cube)
        cube_points=[]
        if(filter_params==None):
            for cube in cube_path[1:]:
                enc_cube=self.encode_cube(cube)
                left,right=self.index_lookup[enc_cube]
                cube_points.append(self.sorted_points[left:right])
        else:
            for cube in cube_path[1:]:
                enc_cube=self.encode_cube(cube)
                left,right=self.index_lookup[enc_cube]
                params=filter_params(self.sorted_points[left:right],obs)
                print(params.shape)
                cube_points.append(params)
            
        distance_info=self.distance_helper(cube_points,start_config)
        print(len(distance_info))
        
        if(len(distance_info)==0):
            print('no info')
            return
        final_path=[]
        min_indices,min_dis=distance_info.pop()
        arg_min=np.argmin(min_dis)
        points=cube_points.pop()
        final_path.append(points[arg_min])
        min_index=min_indices[arg_min]
        while cube_points:
             points=cube_points.pop()
             final_path=[points[min_index]]+final_path
             min_indices,min_dis=distance_info.pop()
             min_index=min_indices[min_index]
        return [start_config]+final_path
    
    def multi_step_path(self, start_config, stop_list, delta=1):
        start_point=robot(start_config/100,[1.0,1.0,1.0])*100
        start_cube=np.floor(start_point/delta)
        cubes=[]
        cubes.append(start_cube)
        current_start=start_cube
        current_end=None
        for i in range(len(stop_list)):
            temp=stop_list[i]
            current_end=np.floor(temp/delta)
            partial_cube_path=self.find_cube_path(current_start, current_end)
            for cube in partial_cube_path[1:]:
                cubes.append(cube)
            current_start=current_end
        configurations=[]
        for c in cubes[1:]:
            enc = self.encode_cube(c)
            left,right=self.index_lookup[enc]
            configurations.append(self.sorted_points[left:right])
        distance_info=self.distance_helper(configurations, start_config)
        
        print(len(distance_info))
        
        if(len(distance_info)==0):
            print('no info')
            return
        final_path=[]
        min_indices,min_dis=distance_info.pop()
        arg_min=np.argmin(min_dis)
        points=configurations.pop()
        final_path.append(points[arg_min])
        min_index=min_indices[arg_min]
        while configurations:
             points=configurations.pop()
             final_path=[points[min_index]]+final_path
             min_indices,min_dis=distance_info.pop()
             min_index=min_indices[min_index]
        return [start_config]+final_path
      
from time import time         
p=Planner()
start=time()
path=p.basic_path(np.array([-0.01569,-0.027759,-0.013276,-0.027759,-0.010862,-0.027759])*100,np.array([15,15,25]),delta=1)
print(time()-start)        
#with open('path.pickle','wb') as handle:
#    pickle.dump(path,handle)        
#p=Planner()
#cubes=p.find_cube_path([-15,13,9],[15,15,25]) 
