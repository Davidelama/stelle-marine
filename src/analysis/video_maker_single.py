import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from celluloid import Camera
import sys
import os
parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)
from single import IO
import json

with open('single_parameters.json',"r") as f:
    details = json.load(f)


nskip=1#per video di tutto usa 200
time_limit=1
initial_skip=0
diam=15
big_diam=details['r_conf']*diam*2
L=big_diam*1.1
Cx=0
Cy=0
dt=0.01

size_cmp=20
norm = mpl.colors.Normalize(vmin=0, vmax=size_cmp)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.hsv)
cmap.set_array([])

name="../../data/01_raw/single/"+IO.get_name(details)+"/"+IO.get_name(details)+".dat.lammpstrj"
print(name)
print("Reading data:")

traj = IO.reconstruct_traj([name], cols=('at_id', 'type', 'x', 'y','q1','q4'))
traj[["x","y"]]*=diam
traj["theta"]=np.arctan2(traj["q4"],traj["q1"])*2
#plt.hist(traj.theta, range=(-np.pi, np.pi), bins=np.linspace(-np.pi, np.pi, 20))
#plt.show()


fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
camera = Camera(fig)
ax.set_xlabel("x[mm]",fontsize=20)
ax.set_ylabel("y[mm]",fontsize=20)
ax.set_aspect(1)
ax.set_xlim(Cx-L/2,Cx+L/2)
ax.set_ylim(Cy-L/2,Cy+L/2)
ax.tick_params(axis='both', which='major', labelsize=20)
s = ((diam*ax.get_window_extent().width  / L * 72./fig.dpi) ** 2)
S=((big_diam*ax.get_window_extent().width  / L * 72./fig.dpi) ** 2)


print("Filming:")
for g, data in traj.groupby(["timestep"]):

    ax.scatter(Cx,Cy,s=S,edgecolors= "blue",color="lightblue")
    ax.scatter(data.x,data.y,s=s,edgecolors= "white",color="k")
    ax.text(0.01,.95,str(g[0]*dt)+r"$\tau$",color="k",transform=ax.transAxes,fontsize=20)
    ax.scatter(data.x + np.cos(data.theta) * diam * .25,
                    data.y + np.sin(data.theta) * diam * .25, s=s * .09, color="white")

    #if i!=0 and i*nskip!=len(t)-1:
    #        vtot=np.sqrt((X[:,(i*nskip+1)]-X[:,(i*nskip-1)])**2+(Y[:,(i*nskip+1)]-Y[:,(i*nskip-1)])**2)
    #        ax.quiver(X[:,i*nskip],Y[:,i*nskip],(X[:,(i*nskip+1)]-X[:,(i*nskip-1)])/vtot,(Y[:,(i*nskip+1)]-Y[:,(i*nskip-1)])/vtot,scale=s/10,color="g")
    #    ax.quiver(X[:,i*nskip],Y[:,i*nskip],np.cos(Phi[:,i*nskip]),np.sin(Phi[:,i*nskip]),scale=s/10,color="r")
    camera.snap()
        

    

animation = camera.animate(80) #the argument is ms/frame
animation.save("../../data/03_analyzed/video/"+IO.get_name(details)+".mp4")

        