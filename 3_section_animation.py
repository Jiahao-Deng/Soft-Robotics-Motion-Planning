from copy import deepcopy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d as p3
import pickle
import os
from numba import jit
os.chdir('C:\\Users\\cdm930PC_7\Desktop\\masterthesis\\3-sec')
with open('path.pickle','rb') as handler:
    path=pickle.load(handler)

#@jit()
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
#@jit()
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
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim3d([0.0, 50])
ax.set_xlabel('X')
ax.set_ylim3d([0.0, 50])
ax.set_ylabel('Y')
ax.set_zlim3d([0.0, 50])
ax.set_zlabel('Z')
point_list=[]
for point in path:
    point_list.append(robot(point/100))
    
line = ax.plot([], [], [])
def animate(i, point_list):
    point_list[i]


line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                              interval=50, blit=False)

plt.show()








def update_lines(num, dataLines, lines) :
    for line, data in zip(lines, dataLines) :
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2,:num])
    return lines
