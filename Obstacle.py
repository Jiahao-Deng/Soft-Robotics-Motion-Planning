# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 08:47:23 2018

@author: cdm930PC_7
"""
import numpy as np
import math
from numba import jit
from path_planning import Planner,encode_cube,decode_cube
import pickle
with open('cube_lookup.pickle','rb') as handle:
    cube_lookup=pickle.load(handle)
jit(nopython=True)
def vectorize_robot(p,xi_list=[1.0,1.0,1.0]):
    L=15/100
    r=1.25/100
    sqrt_3=math.sqrt(3)
    a=p[:,0]
    b=p[:,1]
    xi=xi_list[0]
    a1= 1/2 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 4 * (a + b) / r ** 10 * xi ** 10 / 3444525 + (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 3 * (a + b) / r ** 8 * xi ** 8 / 51030 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 2 * (a + b) / r ** 6 * xi ** 6 / 1215 + (2 * a - b) * (a ** 2 - a * b + b ** 2) * (a + b) / r ** 4 * xi ** 4 / 54 - (2 * a - b) * (a + b) / r ** 2 * xi ** 2 / 6
    a2=(2 * a - b) * (a ** 2 - a * b + b ** 2) ** 4 * (a - b) * sqrt_3 / r ** 10 * xi ** 10 / 3444525 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 3 * (a - b) * sqrt_3 / r ** 8 * xi ** 8 / 51030 + (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 2 * (a - b) * sqrt_3 / r ** 6 * xi ** 6 / 1215 - (2 * a - b) * (a ** 2 - a * b + b ** 2) * (a - b) * sqrt_3 / r ** 4 * xi ** 4 / 54 + (2 * a - b) * (a - b) * sqrt_3 / r ** 2 * xi ** 2 / 6 - sqrt_3/ 2
    a3= -2 / 688905 * (a ** 2 - a * b + b ** 2) ** 4 * (2 * a - b) * sqrt_3 / r ** 9 * xi ** 9 + 0.4e1 / 0.25515e5 * (a ** 2 - a * b + b ** 2) ** 3 * (2 * a - b) * sqrt_3/ r ** 7 * xi ** 7 - 0.2e1 / 0.405e3 * (a ** 2 - a * b + b ** 2) ** 2 * (2 * a - b) * sqrt_3 / r ** 5 * xi ** 5 + 0.2e1 / 0.27e2 * (a ** 2 - a * b + b ** 2) * (2 * a - b) * sqrt_3 / r ** 3 * xi ** 3 - (2 * a - b) * sqrt_3/ r * xi / 3
    a4=-(2 * a - b) * xi ** 2 / r * L * sqrt_3 / 6 + (2 * a - b) * (a ** 2 - a * b + b ** 2) * xi ** 4 / r ** 3 * L * sqrt_3 / 54 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / r ** 5 * L * sqrt_3 / 1215 + (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / r ** 7 * L * sqrt_3 / 51030 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 4 * xi ** 10 / r ** 9 * L * sqrt_3 / 3444525
    a5= -b * (a + b) * (a ** 2 - a * b + b ** 2) ** 4 * sqrt_3 / r ** 10 * xi ** 10 / 3444525 + b / r ** 8 * (a + b) * (a ** 2 - a * b + b ** 2) ** 3 * sqrt_3 * xi ** 8 / 51030 - b / r ** 6 * (a + b) * (a ** 2 - a * b + b ** 2) ** 2 * sqrt_3 * xi ** 6 / 1215 + sqrt_3 * b * (a ** 2 - a * b + b ** 2) * (a + b) / r ** 4 * xi ** 4 / 54 - b / r ** 2 * (a + b) * sqrt_3 * xi ** 2 / 6 + sqrt_3 / 2
    a6=1 / 2 + b * (a - b) * (a ** 2 - a * b + b ** 2) ** 4 / r ** 10 * xi ** 10 / 1148175 - b / r ** 8 * (a - b) * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / 17010 + b / r ** 6 * (a - b) * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / 405 - b / r ** 4 * (a - b) * (a ** 2 - a * b + b ** 2) * xi ** 4 / 18 + b / r ** 2 * (a - b) * xi ** 2 / 2
    a7=-0.2e1 / 0.229635e6 * (a ** 2 - a * b + b ** 2) ** 4 * b / r ** 9 * xi ** 9 + 0.4e1 / 0.8505e4 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 7 * b * xi ** 7 - 0.2e1 / 0.135e3 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 5 * b * xi ** 5 + 0.2e1 / 0.9e1 * (a ** 2 - a * b + b ** 2) / r ** 3 * b * xi ** 3 - b / r * xi
    a8=-L * b / r * xi ** 2 / 2 + b * (a ** 2 - a * b + b ** 2) * xi ** 4 / r ** 3 * L / 18 - b * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / r ** 5 * L / 405 + b * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / r ** 7 * L / 17010 - b * (a ** 2 - a * b + b ** 2) ** 4 * xi ** 10 / r ** 9 * L / 1148175
    a9=0.2e1 / 0.688905e6 * (a ** 2 - a * b + b ** 2) ** 4 * (a + b) * sqrt_3 / r ** 9 * xi ** 9 - 0.4e1 / 0.25515e5 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 7 * (a + b) * sqrt_3 * xi ** 7 + 0.2e1 / 0.405e3 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 5 * (a + b) * sqrt_3 * xi ** 5 - 0.2e1 / 0.27e2 * (a ** 2 - a * b + b ** 2) / r ** 3 * (a + b) * sqrt_3 * xi ** 3 + 1 / r * (a + b) * sqrt_3 * xi / 3
    a10=-0.2e1 / 0.229635e6 * (a ** 2 - a * b + b ** 2) ** 4 * (a - b) / r ** 9 * xi ** 9 + 0.4e1 / 0.8505e4 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 7 * (a - b) * xi ** 7 - 0.2e1 / 0.135e3 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 5 * (a - b) * xi ** 5 + 0.2e1 / 0.9e1 * (a ** 2 - a * b + b ** 2) / r ** 3 * (a - b) * xi ** 3 - 1 / r * (a - b) * xi
    a11=1 - 0.2e1 / 0.3e1 * xi ** 2 * (a ** 2 - a * b + b ** 2) / r ** 2 + 0.2e1 / 0.27e2 * xi ** 4 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 4 - 0.4e1 / 0.1215e4 * xi ** 6 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 6 + 0.2e1 / 0.25515e5 * xi ** 8 * (a ** 2 - a * b + b ** 2) ** 4 / r ** 8 - 0.4e1 / 0.3444525e7 * xi ** 10 * (a ** 2 - a * b + b ** 2) ** 5 / r ** 10
    a12= xi * L - 0.2e1 / 0.9e1 * xi ** 3 * (a ** 2 - a * b + b ** 2) / r ** 2 * L + 0.2e1 / 0.135e3 * xi ** 5 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 4 * L - 0.4e1 / 0.8505e4 * xi ** 7 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 6 * L + 0.2e1 / 0.229635e6 * xi ** 9 * (a ** 2 - a * b + b ** 2) ** 4 / r ** 8 * L
    
    
    xi=xi_list[1]
    if(xi==0):
        x=a4.reshape(-1,1)
        y=a8.reshape(-1,1)
        z=a12.reshape(-1,1)
        return np.concatenate((x,y,z),axis=1)
    a=p[:,2]
    b=p[:,3]
    b1= 1/2 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 4 * (a + b) / r ** 10 * xi ** 10 / 3444525 + (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 3 * (a + b) / r ** 8 * xi ** 8 / 51030 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 2 * (a + b) / r ** 6 * xi ** 6 / 1215 + (2 * a - b) * (a ** 2 - a * b + b ** 2) * (a + b) / r ** 4 * xi ** 4 / 54 - (2 * a - b) * (a + b) / r ** 2 * xi ** 2 / 6
    b2=(2 * a - b) * (a ** 2 - a * b + b ** 2) ** 4 * (a - b) * sqrt_3 / r ** 10 * xi ** 10 / 3444525 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 3 * (a - b) * sqrt_3 / r ** 8 * xi ** 8 / 51030 + (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 2 * (a - b) * sqrt_3 / r ** 6 * xi ** 6 / 1215 - (2 * a - b) * (a ** 2 - a * b + b ** 2) * (a - b) * sqrt_3 / r ** 4 * xi ** 4 / 54 + (2 * a - b) * (a - b) * sqrt_3 / r ** 2 * xi ** 2 / 6 - sqrt_3/ 2
    b3= -2 / 688905 * (a ** 2 - a * b + b ** 2) ** 4 * (2 * a - b) * sqrt_3 / r ** 9 * xi ** 9 + 0.4e1 / 0.25515e5 * (a ** 2 - a * b + b ** 2) ** 3 * (2 * a - b) * sqrt_3/ r ** 7 * xi ** 7 - 0.2e1 / 0.405e3 * (a ** 2 - a * b + b ** 2) ** 2 * (2 * a - b) * sqrt_3 / r ** 5 * xi ** 5 + 0.2e1 / 0.27e2 * (a ** 2 - a * b + b ** 2) * (2 * a - b) * sqrt_3 / r ** 3 * xi ** 3 - (2 * a - b) * sqrt_3/ r * xi / 3
    b4=-(2 * a - b) * xi ** 2 / r * L * sqrt_3 / 6 + (2 * a - b) * (a ** 2 - a * b + b ** 2) * xi ** 4 / r ** 3 * L * sqrt_3 / 54 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / r ** 5 * L * sqrt_3 / 1215 + (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / r ** 7 * L * sqrt_3 / 51030 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 4 * xi ** 10 / r ** 9 * L * sqrt_3 / 3444525
    b5= -b * (a + b) * (a ** 2 - a * b + b ** 2) ** 4 * sqrt_3 / r ** 10 * xi ** 10 / 3444525 + b / r ** 8 * (a + b) * (a ** 2 - a * b + b ** 2) ** 3 * sqrt_3 * xi ** 8 / 51030 - b / r ** 6 * (a + b) * (a ** 2 - a * b + b ** 2) ** 2 * sqrt_3 * xi ** 6 / 1215 + sqrt_3 * b * (a ** 2 - a * b + b ** 2) * (a + b) / r ** 4 * xi ** 4 / 54 - b / r ** 2 * (a + b) * sqrt_3 * xi ** 2 / 6 + sqrt_3 / 2
    b6=1 / 2 + b * (a - b) * (a ** 2 - a * b + b ** 2) ** 4 / r ** 10 * xi ** 10 / 1148175 - b / r ** 8 * (a - b) * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / 17010 + b / r ** 6 * (a - b) * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / 405 - b / r ** 4 * (a - b) * (a ** 2 - a * b + b ** 2) * xi ** 4 / 18 + b / r ** 2 * (a - b) * xi ** 2 / 2
    b7=-0.2e1 / 0.229635e6 * (a ** 2 - a * b + b ** 2) ** 4 * b / r ** 9 * xi ** 9 + 0.4e1 / 0.8505e4 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 7 * b * xi ** 7 - 0.2e1 / 0.135e3 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 5 * b * xi ** 5 + 0.2e1 / 0.9e1 * (a ** 2 - a * b + b ** 2) / r ** 3 * b * xi ** 3 - b / r * xi
    b8=-L * b / r * xi ** 2 / 2 + b * (a ** 2 - a * b + b ** 2) * xi ** 4 / r ** 3 * L / 18 - b * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / r ** 5 * L / 405 + b * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / r ** 7 * L / 17010 - b * (a ** 2 - a * b + b ** 2) ** 4 * xi ** 10 / r ** 9 * L / 1148175
    b9=0.2e1 / 0.688905e6 * (a ** 2 - a * b + b ** 2) ** 4 * (a + b) * sqrt_3 / r ** 9 * xi ** 9 - 0.4e1 / 0.25515e5 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 7 * (a + b) * sqrt_3 * xi ** 7 + 0.2e1 / 0.405e3 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 5 * (a + b) * sqrt_3 * xi ** 5 - 0.2e1 / 0.27e2 * (a ** 2 - a * b + b ** 2) / r ** 3 * (a + b) * sqrt_3 * xi ** 3 + 1 / r * (a + b) * sqrt_3 * xi / 3
    b10=-0.2e1 / 0.229635e6 * (a ** 2 - a * b + b ** 2) ** 4 * (a - b) / r ** 9 * xi ** 9 + 0.4e1 / 0.8505e4 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 7 * (a - b) * xi ** 7 - 0.2e1 / 0.135e3 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 5 * (a - b) * xi ** 5 + 0.2e1 / 0.9e1 * (a ** 2 - a * b + b ** 2) / r ** 3 * (a - b) * xi ** 3 - 1 / r * (a - b) * xi
    b11=1 - 0.2e1 / 0.3e1 * xi ** 2 * (a ** 2 - a * b + b ** 2) / r ** 2 + 0.2e1 / 0.27e2 * xi ** 4 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 4 - 0.4e1 / 0.1215e4 * xi ** 6 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 6 + 0.2e1 / 0.25515e5 * xi ** 8 * (a ** 2 - a * b + b ** 2) ** 4 / r ** 8 - 0.4e1 / 0.3444525e7 * xi ** 10 * (a ** 2 - a * b + b ** 2) ** 5 / r ** 10
    b12= xi * L - 0.2e1 / 0.9e1 * xi ** 3 * (a ** 2 - a * b + b ** 2) / r ** 2 * L + 0.2e1 / 0.135e3 * xi ** 5 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 4 * L - 0.4e1 / 0.8505e4 * xi ** 7 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 6 * L + 0.2e1 / 0.229635e6 * xi ** 9 * (a ** 2 - a * b + b ** 2) ** 4 / r ** 8 * L
    
    t1=a1*b1+a2*b5+a3*b9
    t2=a1*b2+a2*b6+a3*b10
    t3=a1*b3+a2*b7+a3*b11
    t4=a1*b4+a2*b8+a3*b12+a4
    t5=a5*b1+a6*b5+a7*b9
    t6=a5*b2+a6*b6+a7*b10
    t7=a5*b3+a6*b7+a7*b11
    t8=a5*b4+a6*b8+a7*b12+a8
    t9=a9*b1+a10*b5+a11*b9
    t10=a9*b2+a10*b6+a11*b10
    t11=a9*b3+a10*b7+a11*b11
    t12=a9*b4+a10*b8+a11*b12+a12
    
   
    xi=xi_list[2]
    if(xi==0):
        x=t4.reshape(-1,1)
        y=t8.reshape(-1,1)
        z=t12.reshape(-1,1)
        return np.concatenate((x,y,z),axis=1)
    a=p[:,4]
    b=p[:,5]
    c4=-(2 * a - b) * xi ** 2 / r * L * sqrt_3 / 6 + (2 * a - b) * (a ** 2 - a * b + b ** 2) * xi ** 4 / r ** 3 * L * sqrt_3 / 54 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / r ** 5 * L * sqrt_3 / 1215 + (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / r ** 7 * L * sqrt_3 / 51030 - (2 * a - b) * (a ** 2 - a * b + b ** 2) ** 4 * xi ** 10 / r ** 9 * L * sqrt_3 / 3444525
    c8=-L * b / r * xi ** 2 / 2 + b * (a ** 2 - a * b + b ** 2) * xi ** 4 / r ** 3 * L / 18 - b * (a ** 2 - a * b + b ** 2) ** 2 * xi ** 6 / r ** 5 * L / 405 + b * (a ** 2 - a * b + b ** 2) ** 3 * xi ** 8 / r ** 7 * L / 17010 - b * (a ** 2 - a * b + b ** 2) ** 4 * xi ** 10 / r ** 9 * L / 1148175
    c12= xi * L - 0.2e1 / 0.9e1 * xi ** 3 * (a ** 2 - a * b + b ** 2) / r ** 2 * L + 0.2e1 / 0.135e3 * xi ** 5 * (a ** 2 - a * b + b ** 2) ** 2 / r ** 4 * L - 0.4e1 / 0.8505e4 * xi ** 7 * (a ** 2 - a * b + b ** 2) ** 3 / r ** 6 * L + 0.2e1 / 0.229635e6 * xi ** 9 * (a ** 2 - a * b + b ** 2) ** 4 / r ** 8 * L
    x=(t1*c4+t2*c8+t3*c12+t4).reshape(-1,1)
    y=(t5*c4+t6*c8+t7*c12+t8).reshape(-1,1)
    z=(t9*c4+t10*c8+t11*c12+t12).reshape(-1,1)
    return np.concatenate((x,y,z),axis=1)

def filter_cubes(obstacle):
    cube_sets=set()
    for o in obstacle:
        center=np.array(o[0:3])
        r=o[3]
        a_low=math.floor((center[0]-r)/1)
        a_high=math.floor((center[0]+r)/1)
        b_low=math.floor((center[1]-r)/1)
        b_high=math.floor((center[1]+r)/1)
        c_low=math.floor((center[2]-r)/1)
        c_high=math.floor((center[2]+r)/1)
        X,Y,Z = np.mgrid[a_low:a_high+1:1,b_low:b_high+1:1,c_low:c_high+1:1]
        xyz = np.vstack((X.flatten(), Y.flatten(),Z.flatten())).T
        distances=np.linalg.norm((xyz+0.5-center),axis=1)
        delete_cubes=xyz[distances<(0.87+r)]
        
        for cube in delete_cubes:
            encoding=encode_cube(cube)
            index=cube_lookup[encoding]
            if(index not in cube_sets):
                cube_sets.add(index)
                print(index)
    return list(cube_sets)
        

def filter_params(p,obstacle):
    num_sec=3
    step=0.0
    sec_number=1
    precision=0.1
    while(sec_number <= num_sec):
        xi_list=[1.0]*(sec_number-1)+[step]+[0.0]*(num_sec-sec_number)
       
        for o in obstacle:
            center=np.array(o[0:3])
            r=o[3]
            result=vectorize_robot(p/100,xi_list)
            result =result*100
            distances=np.linalg.norm(result-center,axis=1)
            p=p[distances>r]
        if(p.shape[0]==0):
            print('this cube falied')
            return
        if(step>=1):
            sec_number +=1
            step=0.0
            continue
        step +=precision
    return p
        
   
def obstacle(obs,start_config,end_point):
    p=Planner()
    deleted_cubes=filter_cubes(obs)
    edges=[]
    for cube in deleted_cubes:
        inci=p.cube_graph.incident(cube)
        edges += inci
    edges=list(set(edges))
    p.cube_graph.delete_edges(edges)
    path=p.basic_path(start_config,end_point,filter_params,obs,1)
    del p
    return path
        

obs=[[-3,13,12,3],[-8,20,12,3],[10,13,12,3]]
path=obstacle(obs,np.array([-0.01569,-0.027759,-0.013276,-0.027759,-0.010862,-0.027759])*100,np.array([15,15,25]))
