import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

def deg2rad(deg):
    rad = (deg/360)*2*np.pi
    return rad

def rad2deg(rad):
    deg = (rad/(2*np.pi)) * 360
    return deg


# Define theta and phi in radians
theta=  np.array([-0.06682738,  1.1693791,   0.27224414])
phi=  np.array([-0.82274821,  0.44785149,  0.38659043])

# Define the fixed variables
alpha = 2
zeta = deg2rad(np.array([64, 55.3, 48.8]))
l = [0.05, 0.03, 0.02]
r = [0.03, 0.06, 0.05, 0.053, 0.016, 0.014]
A = np.array([-0.015, -0.06])

origin = np.array([0,0])


# Calculate the finger points
fpoint1 = origin + np.array([l[0]*np.cos(theta[0]), l[0]*np.sin(theta[0])])
fpoint2 = fpoint1 + np.array([l[1]*np.cos(theta[0]+theta[1]), l[1]*np.sin(theta[0]+theta[1])])
fpoint3 = fpoint2 + np.array([l[2]*np.cos(theta[0]+theta[1]+theta[2]), l[2]*np.sin(theta[0]+theta[1]+theta[2])])
fpoints = [origin, fpoint1, fpoint2, fpoint3]

# Calulcate the rf points
rfpoint1 = A

rfpoint2 = rfpoint1 + np.array([r[0]*np.cos(phi[0]), r[0]*np.sin(phi[0])])
rfpoint3 = rfpoint2 + np.array([r[1]*np.cos(phi[0]+zeta[0]), r[1]*np.sin(phi[0]+zeta[0])])
rfpoint4 = rfpoint3 + np.array([r[2]*np.cos(phi[0]+phi[1]+zeta[0]), r[2]*np.sin(phi[0]+phi[1]+zeta[0])])
rfpoint5 = rfpoint4 + np.array([r[3]*np.cos(phi[0]+phi[1]+zeta[0]+zeta[1]), r[3]*np.sin(phi[0]+phi[1]+zeta[0]+zeta[1])])
rfpoint6 = rfpoint5 + np.array([r[4]*np.cos(phi[0]+phi[1]+zeta[0]+zeta[1]+zeta[2]), r[4]*np.sin(phi[0]+phi[1]+zeta[0]+zeta[1]+zeta[2])])
rfpoint7 = rfpoint6 + np.array([r[5]*np.cos(phi[0]+phi[1]+phi[2]+zeta[0]+zeta[1]+zeta[2]), r[5]*np.sin(phi[0]+phi[1]+phi[2]+zeta[0]+zeta[1]+zeta[2])])
rfpoints = [rfpoint1, rfpoint2, rfpoint3, rfpoint4, rfpoint5, rfpoint6, rfpoint7]

# Get line segments
line_collection = []
for i in range(len(fpoints)-1):
    line_collection.append([fpoints[i], fpoints[i+1]])
for i in range(len(rfpoints)-1):
    line_collection.append([rfpoints[i], rfpoints[i+1]])
line_collection = mc.LineCollection(line_collection, colors='k', linewidth=2)

# Code to make a figure, plot the points and line segments
fig, ax = plt.subplots()

for p in fpoints:
    ax.scatter(p[0], p[1], c='b')
for p in rfpoints:
    ax.scatter(p[0], p[1], c='r')



ax.add_collection(line_collection)
ax.set_ylim(-1,1)
ax.set_xlim(-1,1)
plt.show()

