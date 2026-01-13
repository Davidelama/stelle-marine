import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from celluloid import Camera
import sys
import os
parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)
from stelle import IO
import json

def cmapper(max, cm=mpl.cm.plasma,cmap_mod=1.2):
    norm = mpl.colors.Normalize(vmin=0, vmax=max*cmap_mod-1)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    cmap.set_array([])
    return cmap

with open('stelle_parameters.json',"r") as f:
    details = json.load(f)


nskip=1#per video di tutto usa 200
time_limit=1
initial_skip=0

diam=15
if details['r_conf']!=0:
    diam_core=details["r_core"]*2*diam
    big_diam=details['r_conf']*diam*2
    L=big_diam*1.1
    Cx=0
    Cy=0
if details['brownian']==0:
    dt=min(0.001,0.01/details["gamma"])
else:
    dt=0.0001

cmap = cmapper(details["functionality"]*details["n_mol"], cm=mpl.cm.jet,cmap_mod=1)

name="../../data/01_raw/stelle/"+IO.get_name(details)+"/"+IO.get_name(details)+".dat.lammpstrj"
print(name)
print("Reading data:")

traj = IO.reconstruct_traj([name], cols=('at_id', 'mol_id', 'type', 'x', 'y','mux','muy'))
traj[["x","y"]]*=diam
traj["theta"]=np.arctan2(traj["muy"],traj["mux"])*2
ntot=(1+details["functionality"]*(details["n_beads"]+1))*details["n_mol"]
traj["p_idx"]=((np.arange(len(traj))%ntot)-traj["mol_id"])//(details["n_beads"]+1)
traj.loc[traj.type==1,"p_idx"]=-1

#plt.hist(traj.theta, range=(-np.pi, np.pi), bins=np.linspace(-np.pi, np.pi, 20))
#plt.show()


fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
camera = Camera(fig)
ax.set_xlabel("x[mm]",fontsize=20)
ax.set_ylabel("y[mm]",fontsize=20)
ax.set_aspect(1)
ax.tick_params(axis='both', which='major', labelsize=20)
if details['r_conf']!=0:
    ax.set_xlim(Cx-L/2,Cx+L/2)
    ax.set_ylim(Cy-L/2,Cy+L/2)
    s = ((diam*ax.get_window_extent().width / L * 72./fig.dpi) ** 2)
    s2 = ((diam * ax.get_window_extent().width * .2 / L * 72. / fig.dpi) ** 2)
    s_core = ((diam_core*ax.get_window_extent().width  / L * 72./fig.dpi) ** 2)
    S=((big_diam*ax.get_window_extent().width  / L * 72./fig.dpi) ** 2)


print("Filming:")
for g, data in traj.groupby(["timestep"]):
    if details['r_conf'] != 0:
        ax.scatter(Cx,Cy,s=S,edgecolors= "blue",color="lightblue")
        ax.scatter(data.x[data.type == 1], data.y[data.type == 1], s=s_core, edgecolors="k", color="w",lw=2)
        for f in range(0,details["functionality"]*details["n_mol"]):
            ax.plot(data.x[data.p_idx == f], data.y[data.p_idx == f], color=cmap.to_rgba(f),lw=2)
        ax.scatter(data.x[data.type == 2], data.y[data.type == 2], s=s2, edgecolors="k", color=cmap.to_rgba(data.p_idx[data.type==2]))
        ax.scatter(data.x[data.type==3],data.y[data.type==3],s=s,edgecolors= cmap.to_rgba(data.p_idx[data.type==3]),color="w",lw=2)
        ax.scatter(data.x[data.type == 3] + np.cos(data.theta[data.type == 3]) * diam * .25,
                   data.y[data.type == 3] + np.sin(data.theta[data.type == 3]) * diam * .25, s=s * .09, color="k")
    else:
        ax.scatter(data.x[data.type == 1], data.y[data.type == 1], edgecolors="k", color="w",s=50)
        for f in range(0,details["functionality"]*details["n_mol"]):
            ax.plot(data.x[data.p_idx == f], data.y[data.p_idx == f], color=cmap.to_rgba(f),lw=2)
        ax.scatter(data.x[data.type == 2], data.y[data.type == 2], edgecolors="k",
                   color=cmap.to_rgba(data.p_idx[data.type == 2]), s=3)
        ax.scatter(data.x[data.type == 3], data.y[data.type == 3], edgecolors="k", color=cmap.to_rgba(data.p_idx[data.type == 3]))
    ax.text(0.01,.95,str(g[0]*dt)+r"$\tau$",color="k",transform=ax.transAxes,fontsize=20)

    #if i!=0 and i*nskip!=len(t)-1:
    #        vtot=np.sqrt((X[:,(i*nskip+1)]-X[:,(i*nskip-1)])**2+(Y[:,(i*nskip+1)]-Y[:,(i*nskip-1)])**2)
    #        ax.quiver(X[:,i*nskip],Y[:,i*nskip],(X[:,(i*nskip+1)]-X[:,(i*nskip-1)])/vtot,(Y[:,(i*nskip+1)]-Y[:,(i*nskip-1)])/vtot,scale=s/10,color="g")
    #    ax.quiver(X[:,i*nskip],Y[:,i*nskip],np.cos(Phi[:,i*nskip]),np.sin(Phi[:,i*nskip]),scale=s/10,color="r")
    camera.snap()
        

    

animation = camera.animate(80) #the argument is ms/frame
animation.save("../../data/03_analyzed/video/"+IO.get_name(details)+".mp4")

        