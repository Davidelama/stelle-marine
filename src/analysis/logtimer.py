import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#regola del pollice: tot/dump dovrebbe essere almeno 100
dump=1e4
tot=1e6
exponent=int(np.log2(dump))
edump=int(2**exponent)
nedump=int(tot//edump)
ntot=int(nedump*(exponent+1)+1)

tvec=np.zeros(ntot)
tarr=2**np.arange(exponent+1)
tvec[0:exponent+1]=tarr
for i in range(1,nedump):
    tvec[i*(exponent+1):(i+1)*(exponent+1)]=tarr+tvec[i*(exponent+1)-1]
tvec[-1]=tot+1
np.savetxt('logtime.txt',tvec, fmt='%d')


fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
ax.set_xlabel("t",fontsize=20)
ax.set_ylabel("MSD",fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)

ax.plot(tvec,lw=2,c="k")
 
#ax.set_xscale("log")
#ax.set_yscale("log")
fig.savefig("tlog.png", bbox_inches="tight",dpi=300)
        