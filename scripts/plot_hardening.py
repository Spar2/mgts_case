"""Publication-style figures from the fixed experiment and its counterexample."""
import json
import os
import sys
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'artifacts/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fiber.classify.physical import FEATURE_NAMES,shape_features

out=ROOT/'docs/images/hardening'
out.mkdir(parents=True,exist_ok=True)
times=np.arange(0,3600,60.)
x=np.arange(80.)
fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
for ax,periodic in zip(axes,(True,False),strict=True):
    field=np.full((len(times),len(x)),22.)
    core=(x>=38)&(x<44)
    active=((times>=120)&(((times-120)%600)<120)) if periodic else ((times>=120)&(times<840))
    field[np.ix_(active,core)]+=4
    features=dict(zip(FEATURE_NAMES,shape_features(field,np.ones_like(field,bool),x,times),strict=True))
    ax.step(times/60,field[:,40]-22,where='post',color='#176c83',linewidth=2)
    ax.set_title(f"{'Repeated pulses' if periodic else 'Continuous block'}: episodes={features['positive_episodes']:.0f}, period={features['period_s']:.0f} s, duty={features['positive_duty']:.2f}",loc='left')
    ax.set_ylabel('Contrast, °C')
    ax.grid(alpha=.2)
axes[-1].set_xlabel('Observed time, min')
fig.savefig(out/'temporal_order.png',dpi=170)
plt.close(fig)

report=json.loads((ROOT/'artifacts/hardening/physical/report.json').read_text())
matrix=np.array(report['test']['confusion_matrix'])
names=['Background','Heating','Cooling','Periodic','Broad shift','Single spike*']
fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=64)
for row in range(6):
    for col in range(6):
        ax.text(col,row,str(matrix[row,col]),ha='center',va='center',color='white' if matrix[row,col]>40 else '#172a35')
ax.set_xticks(range(6),names,rotation=35,ha='right')
ax.set_yticks(range(6),names)
ax.set_xlabel('Predicted shape')
ax.set_ylabel('Planted synthetic class')
ax.set_title('Held-out physical scene classification: 384 scenes\n*Sensor_Fault is a synthetic spike pattern, not confirmed hardware failure',fontsize=11)
fig.colorbar(im,ax=ax,label='Number of scenes',shrink=.8)
fig.savefig(out/'physical_confusion.png',dpi=170)
plt.close(fig)
print(out)
