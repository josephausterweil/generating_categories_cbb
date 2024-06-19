import pickle, math
import pandas as pd
import sqlite3
import random
import os
import itertools

import pickle_compat

pickle_compat.patch()

from io import open
def compile_file(filename):
	with open(filename, encoding='utf-8') as f:
		return compile(f.read(), filename, 'exec')

cur_dir = 'Experiments/multiexpt_modeling'

exec(compile_file(os.path.join(cur_dir,'Imports.py')))
exec(compile_file(os.path.join(cur_dir,'ImportModels.py')))

import Modules.Funcs as funcs
from Modules.Classes import Simulation
from Modules.Classes import CopyTweak
from Modules.Classes import CopyTweakRep
from Modules.Classes import Packer
from Modules.Classes import PackerRep
from Modules.Classes import PackerEuc
from Modules.Classes import ConjugateJK13
from Modules.Classes import RepresentJK13
from scipy.stats import stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

#plotting options
STAT_LIMS =  (-1.0, 1.0)
#Fit to only last trial?
fitlast = False
#Make plots?
doplots = False
saveplots = True
if fitlast:
    showlast = True
else:
    showlast = False #True = Show last even if fitting to all?
#Bootstrap parameters
nbootstraps = 1000

#Some plotting options
font = {'family' : 'DejaVu Sans',
        'weight' : 'regular',
        'size'   : 15}

#Specify simulation values
N_SAMPLES = 10000
WT_THETA = 1.5
MIN_LL = 1e-10

# Specify default dataname
dbname = 'experiment-corner.db'
dataname_def = 'corner'

participant_def = 'all'
unique_trials_def = 'all'
dataname = dataname_def
exec(compile_file(os.path.join(cur_dir,'validate_data.py')))

bestparmdb = "pickles/chtc_gs_best_params_{}".format(src)


plt.rc('font', **font)

# get data from pickle
with open(os.path.join(cur_dir,pickledir+src), "rb" ) as f:
    trials = pickle.load( f,encoding='latin1')

# get best params pickle
#bestparmdb = "pickles/chtc_gs_best_params_all_data_e1_e2.p"
#bestparmdb = "pickles/chtc_gs_best_params_corrs.p"
with open(os.path.join(cur_dir,bestparmdb), "rb" ) as f:
    best_params_t = pickle.load( f, encoding='latin1')

#Rebuild it into a smaller dict
best_params = dict()
for modelname in best_params_t.keys():    
    best_params[modelname] = dict()
    for i,parmname in enumerate(best_params_t[modelname]['parmnames']):
        parmval = best_params_t[modelname]['bestparmsll']
        best_params[modelname][parmname] = parmval[i]
modelList = [Packer,RepresentJK13]

#Specify plot order
modelPlotOrder = np.array([[Packer,RepresentJK13],[CopyTweak,ConjugateJK13]])
#modelPlotOrder = np.array([[CopyTweak,CopyTweakRep],[Packer,RepresentJK13]])

        
unique_trials = 'all'
trials.task = task

#Create new trialset
con = sqlite3.connect(os.path.join(cur_dir,dbname))
participants = pd.read_sql_query("SELECT participant, condition from participants", con)
generation = pd.read_sql_query("SELECT * from generation", con)
alphas = pd.read_sql_query("SELECT * from alphas", con)
stimuli = pd.read_sql_query("SELECT * from stimuli", con).values
con.close()


firsts = []
firsts += [generation.loc[(generation.wrap_ax!=1.0) & (generation.trial==0),'stimulus']] #Squares
firsts += [generation.loc[(generation.wrap_ax==1.0) & (generation.trial==0),'stimulus']] #Circles

#Convert to ps
ps = []
f,ax = plt.subplots(1,len(firsts),figsize=(3*len(firsts),3))
A = alphas.Corner_C.values
As = stimuli[A]
step = .25
adj = - step/2
for i,first in enumerate(firsts):
    stimvals = stimuli[first]
    #Add jitter
    stimvals = funcs.jitterize(stimvals,sd=.03)
    ax[i].scatter(stimvals[:,0],stimvals[:,1],alpha=.5)
    stepx = np.unique(stimuli[:,0]) + adj
    stepy = np.unique(stimuli[:,1]) + adj
    stepx = np.append(stepx,max(stepx)+step)
    stepy = np.append(stepy,max(stepy)+step)
    for pi,xx in enumerate(stepx):
        ax[i].grid(False)
        ax[i].plot([xx,xx],[min(stepy),max(stepy)],'k-')        
        ax[i].plot([min(stepx),max(stepx)],[xx,xx],'k-')
        for AsEl in As:
            ax[i].text(AsEl[0],AsEl[1],'A',color='red',
                       horizontalalignment='center',verticalalignment='center',
                      alpha=.15,
                      fontsize=8)

# Probabilities at outer edge vs inner space - Rep

steps = 50
vals = np.linspace(-1, 1,steps).tolist()
space = np.fliplr(funcs.cartesian([vals, vals]))
st = 2./(steps-1)

models = [Packer,RepresentJK13]
f,ax = plt.subplots(2,len(models),figsize=(7,3.2*len(models)))
wraps = [None,1]
labels = [['a','b'],['c','d']]
for wi,wrap_ax in enumerate(wraps):
    for mi,model in enumerate(models):
        #outind = range(steps) + [(e+1)*steps for e in range(steps-2)] + [(e+1)*steps+steps-1 for e in range(steps-2)] + range(steps**2-steps,steps**2)        

        outind = list(range(steps)) + [(e+1)*steps for e in range(steps-2)] + [(e+1)*steps+steps-1 for e in range(steps-2)] + list(range(steps**2-steps,steps**2))

        innind = [i for i in range(steps**2) if not i in outind]
        params = best_params[model.model]
        categories1 = [np.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]])]
        temp1 = model(categories1,params,wrap_ax=wrap_ax).get_generation_ps(space,1,'generate')
        outps = sum(temp1[outind])
        innps = sum(temp1[innind])
        maxout = max(temp1[outind])
        maxin = max(temp1[innind])
        print('{} - Outer edges: {}, Max: {}'.format(model.modelshort,np.round(outps,2),np.round(maxout,4)))
        print('{} - Inner spaces: {}, Max: {}'.format(model.modelshort, np.round(innps,2),np.round(maxin,4)))
        gps = funcs.gradientroll(temp1,'roll')[:,:,0]
        ps_ElRange = gps.max()-gps.min()
        plotVals = (gps-gps.min())/ps_ElRange
        gammas = []
        categories1[0] = [[-.9,-.9],[-.9,.9],[.9,-.9],[.9,.9]]
        funcs.plotgradient(ax[wi,mi], plotVals, categories1[0], [], gammas = gammas,clim = [0,1], cmap = 'Blues',beta_col='green')
        if wi==0:
            modeln = model.modelshort 
            if mi==0:
                ax[wi,mi].set_ylabel('Squares\n\n\n',fontsize=12)
        else:
            modeln = ''
            if mi==0:
                ax[wi,mi].set_ylabel('Circles\n\n(Toroidal)\nOrientation',fontsize=12)
            else:
                ax[wi,mi].set_ylabel('(Toroidal) \n Orientation',fontsize=12)
            ax[wi,mi].set_xlabel('Size\n(Bounded)',fontsize=12)
        
        #titlestr = '%s\nOuter edges: %.2f \nInner spaces: %.2f' % (modeln,outps,innps)
        titlestr = '%s\n\n (%s)'% (modeln,labels[wi][mi])
        ax[wi,mi].set_title(titlestr,fontsize=12)
        
if saveplots:
    plt.savefig(os.path.join(cur_dir,'firstbetas_cornerPostfits50.pdf'),bbox_inches='tight')