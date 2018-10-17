# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:29:54 2018

@author: cdm930PC_7
"""

from copy import deepcopy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
import pickle
import os
from numba import jit
os.chdir('C:\\Users\\cdm930PC_7\Desktop\\masterthesis\\3-sec')
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
    xi1,xi2,xi3=xi_list
    if(xi1<1):
        m=section(p[0],p[1],xi1)
    elif(xi2<1):
        m1=section(p[0],p[1],1)
        m2=section(p[2],p[3],xi2)
        m=m1.dot(m2)
    else:
        m1=section(p[0],p[1],1)
        m2=section(p[2],p[3],1)
        m3=section(p[4],p[5],xi_list[2])
        m=m1.dot(m2)
        m=m.dot(m3)
    return m.T[3][0:3]
        
def points(l,precision=0.05):
    d={}
    x=[]
    y=[]
    z=[]
    num_sec=len(l)//2
    step=0.0
    sec_number=1
    while(sec_number <= num_sec):
        xi=[1.0]*(sec_number-1)+[step]+[0.0]*(num_sec-sec_number)
        result=robot(l,xi)
        result =result*100
        #print(result)
        x.append(result[0])
        y.append(result[1])
        z.append(result[2])
        if(step>=1):
            d[sec_number]=[x,y,z]
            x=deepcopy([])
            y=deepcopy([])
            z=deepcopy([])
            sec_number +=1
            step=0.0
            continue
        step +=precision
    return d

def frame(l1,l2,interval=20):
    step=(l2-l1)/interval
    r=[]
    for b in range(interval):
        d = points(l1)
        r.append(d)
        l1 =l1+step
    return r


def tip(l1,l2,interval=20):
    num_sec=len(l1)//2
    l1=np.array(l1,dtype='float32')
    l2=np.array(l2,dtype='float32')
    step = (l2 - l1) / interval
    x=[]
    y=[]
    z=[]
    xi=[1]*num_sec
    for b in range(interval):
        tip=robot(l1,xi)*100
        x.append(tip[0])
        y.append(tip[1])
        z.append(tip[2])
        l1 += step
    return [x,y,z]



#
    








body=[]
tips=[]
l=[]


for i in range(1,len(path)):
    l1=path[i-1]/100
    l2=path[i]/100
    body += frame(l1,l2)
    t =tip(l1,l2)
    if(len(tips)==0):
        tips=t
    else:
        tips[0] +=t[0]
        tips[1] += t[1]
        tips[2] += t[2]

u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 30)
center=obs[0][0:3]
r=obs[0][3]
x = r*np.outer(np.sin(u), np.sin(v))+center[0]
y = r*np.outer(np.sin(u), np.cos(v))+center[1]
z = r*np.outer(np.cos(u), np.ones_like(v))+center[2]
for i in range(1,3):
    center=obs[i][0:3]
    r=obs[i][3]
    x1=r*np.outer(np.sin(u), np.sin(v))+center[0]
    y1=r*np.outer(np.sin(u), np.cos(v))+center[1]
    z1=r*np.outer(np.cos(u), np.ones_like(v))+center[2]
    x=np.concatenate((x,x1),axis=0)
    y=np.concatenate((y,y1),axis=0)
    z=np.concatenate((z,z1),axis=0)
    
colors=['r','y','g']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
target=[15,15,25]
ax.scatter(np.array(target[0]),np.array(target[1]), np.array(target[2]))
ax.set_zlim(-30,60)
ax.set_xlim(30,-30)
ax.set_ylim(30,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
lines = [ax.plot([], [], [],'-', c=c)[0]
             for c in colors]
tiplist, =ax.plot([],[],[],'-',c='black')
lines.append(tiplist)

def animate(i):
    for a in range(len(lines)-1):
       lines[a].set_xdata(body[i][a+1][0])
       lines[a].set_ydata(body[i][a+1][1])
       lines[a].set_3d_properties(body[i][a+1][2])
    x_tip = tips[0][0:i + 1]
    y_tip = tips[1][0:i + 1]
    z_tip = tips[2][0:i + 1]
    lines[a+1].set_xdata(x_tip)
    lines[a+1].set_ydata(y_tip)
    lines[a+1].set_3d_properties(z_tip)
    # update the data
    return lines
ax.plot_wireframe(x, y, z)

#
#def animatetip(i):
#    x_tip = tips[0][0:i + 1]
#    y_tip = tips[1][0:i + 1]
#    z_tip = tips[2][0:i + 1]
#    tiplist.set_xdata(x_tip)
#    tiplist.set_ydata(y_tip)
#    tiplist.set_3d_properties(z_tip)
#
#    return tiplist

# call the animator.  blit=True means only re-draw the parts that have changed.
# blit=True dose not work on Mac, set blit=False
# interval= update frequency
frame_length=(len(path)-1)*100-1
ani= animation.FuncAnimation(fig=fig, func=animate, frames=frame_length,
                              interval=20, blit=False,repeat=True)


plt.show()

