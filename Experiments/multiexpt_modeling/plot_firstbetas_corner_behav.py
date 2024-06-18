#reproducing plot_1stbetasCornerBehav.ipynb as python script
#Can we get plots of first betas?
#Shows each individual plot along with which model best fits itimport pickle, math
import pickle,math
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

cur_dir = 'Experiments/cogpsych_code'

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
saveplots = False
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
dbname = 'experiment-corner.db'#'experiments-5con.db'#raw data
dataname_def = 'corner'#'5con'#bestparms comes from here

# Specify default dataname
# dataname_def = 'pooled'#'nosofsky1986'#'NGPMG1994'
participant_def = 'all'
unique_trials_def = 'all'
dataname = dataname_def


exec(compile_file(os.path.join(cur_dir,'validate_data.py')))

bestparmdb = "pickles/chtc_gs_best_params_{}".format(src)


plt.rc('font', **font)

# get data from pickle
with open(os.path.join(os.path.join(cur_dir,pickledir),src),'rb') as f:
    trials = pickle.load( f ,encoding='latin1')

# get best params pickle
#bestparmdb = "pickles/chtc_gs_best_params_all_data_e1_e2.p"
#bestparmdb = "pickles/chtc_gs_best_params_corrs.p"

with open(os.path.join(cur_dir,bestparmdb), "rb" ) as f:
    best_params_t = pickle.load( f,encoding='latin1' )

#Rebuild it into a smaller dict
best_params = dict()
for modelname in best_params_t.keys():    
    best_params[modelname] = dict()
    for i,parmname in enumerate(best_params_t[modelname]['parmnames']):
        parmval = best_params_t[modelname]['bestparmsll']
        best_params[modelname][parmname] = parmval[i]
modelList = [Packer,RepresentJK13]
#modelList = [CopyTweak,CopyTweakRep,Packer, RepresentJK13,]                            
#Specify plot order
modelPlotOrder = np.array([[Packer,RepresentJK13],[CopyTweak,ConjugateJK13]])
     
unique_trials = 'all'
trials.task = task

#Create new trialset
con = sqlite3.connect(os.path.join(cur_dir,dbname))
participants = pd.read_sql_query("SELECT participant, condition from participants", con)
generation = pd.read_sql_query("SELECT * from generation", con)
alphas = pd.read_sql_query("SELECT * from alphas", con)
stimuli = pd.read_sql_query("SELECT * from stimuli", con).values
con.close()

#first cell end
#start second cell
firsts = []
firsts += [generation.loc[(generation.wrap_ax!=1.0) & (generation.trial==0),'stimulus']] #Squares
firsts += [generation.loc[(generation.wrap_ax==1.0) & (generation.trial==0),'stimulus']] #Circles
#Convert to ps
ps = []
f,ax = plt.subplots(1,len(firsts),figsize=(3*len(firsts),3))
A = alphas.Corner_C.values
As = stimuli[A]
titles = ['Squares','Circles']
for i,first in enumerate(firsts):
    freq = first.value_counts(normalize=True,sort=False).sort_index()
    p = np.zeros(len(stimuli))
    p[freq.keys()] = freq.values
    ps += [p]    
    gps = funcs.gradientroll(p,'roll')[:,:,0]
    ps_ElRange = gps.max()-gps.min();
    plotVals = (gps-gps.min())/ps_ElRange
    gammas = []
    im = funcs.plotgradient(ax[i], plotVals, As, [], gammas = gammas,clim = [0,1], cmap = 'Blues',beta_col='green')
    ax[i].set_title(titles[i])
# add colorbar
f.subplots_adjust(right=0.8)
cbar = f.add_axes([0.88, 0.17, 0.04, 0.66])
cb = f.colorbar(im, cax=cbar, ticks = [0,1])#,boundaries=np.unique(plotVals))
cbar.set_yticklabels(['Lowest\nFrequency', 'Greatest\nFrequency'],fontsize=12)
# cb = f.colorbar(im, cax=cbar, ticks = np.unique(plotVals),boundaries=np.unique(plotVals))
# cbar.set_yticklabels(range(6),fontsize=10)
cbar.tick_params(length = 0)

plt.show()

if saveplots:
    plt.savefig('private/firstbetas_cornerbehav.pdf',bbox_inches='tight')

#last cell for manuscript
#Get distances from center
from scipy.stats import pearsonr
center = np.array([0,0])
f,ax = plt.subplots(2,2,figsize=(8,8))
models = [Packer,RepresentJK13]#,RepresentJK13]
modelc = ['r','g']
modelm = ['x','o']
ys = ['Squares','Circles']
categories = [np.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]])]
wraps = [None,1]
bwidth = .2 #.1#bar width
fs = 18


for i,first in enumerate(firsts):
    for mi,model in enumerate(models):
        params = best_params[model.model]
        #firsts: squares, then circles, plotted as rows
        wrap_ax = wraps[i]
        stimvals = stimuli[first]
        distances = np.abs(stimvals)
        distx = distances[:,0]
        disty = distances[:,1]
        if mi==0:
            #Normalize them so they sum to 1
            #Bin them manually
            distxu = np.unique(distx)
            distyu = np.unique(disty)
            n_max =  [1]+[2 for _ in range(len(distxu)-1)]#adjust for there being twice as many edges as centers (i.e., there is only one clumn of 0s, but .25,.5,.75, and 1 needs to be averaged)
            nx = np.array([np.count_nonzero(distx==x) for x in distxu],dtype=float)/n_max
            ny = np.array([np.count_nonzero(disty==y) for y in distyu],dtype=float)/n_max
            #Normalise so bins sum to 1
            nxp = nx/sum(nx)
            nyp = ny/sum(ny)
#             ax[i,0].bar(distxu,nxp,width=bwidth,color='None',edgecolor='k')
#             ax[i,1].bar(distyu,nyp,width=bwidth,color='None',edgecolor='k')
            ax[i,0].bar(distxu,nxp,width=bwidth)
            ax[i,1].bar(distyu,nyp,width=bwidth)    
            datarx = pearsonr(nxp,distxu)
            datary = pearsonr(nyp,distyu)
            print('Data %s corr x = %f, p = %f'%(ys[i],datarx[0],datarx[1]))
            print('Data %s corr y = %f, p = %f'%(ys[i],datary[0],datary[1]))
            
#             mx = datarx[0]*np.std(nxp)/np.std(distxu)#np.cov(distxu,nxp)[0,1]
#             cx = np.mean(nxp) - mx*(np.mean(distxu))
    
#             my = datary[0]*np.std(nyp)/np.std(distyu)#np.cov(distxu,nxp)[0,1]
#             cy = np.mean(nyp) - my*(np.mean(distyu))
#             ax[i,0].plot([0,1],[cx,cx+mx],'-')
#             ax[i,1].plot([0,1],[cy,cy+my],'-')

            #Try other distance metrics?
#             distc = np.sum(distances,axis=1) #city block
#             diste = np.sqrt(np.sum(distances**2,axis=1)) #Euclidean
#             datac = pearsonr(distc)
#             print('Data %s corr city = %f, p = %f'%(ys[i],datarc[0],datarc[1]))
#             print('Data %s corr euc = %f, p = %f'%(ys[i],datare[0],datare[1]))

        # Get model predictions
        ps = model(categories,params,wrap_ax=wrap_ax).get_generation_ps(stimuli,1,'generate')                
        distsp = np.abs(stimuli)
        distxp = distsp[:,0]
        distyp = distsp[:,1]
        #Get average ps?
        distxpu = np.unique(distxp)
        distypu = np.unique(distyp)
        psxu = np.array([np.mean(ps[distxp==x]) for x in distxpu])
        psyu = np.array([np.mean(ps[distyp==y]) for y in distypu])
        #Normalize them so they sum to 1
        psxu = np.array(psxu)/np.sum(psxu)
        psyu = np.array(psyu)/np.sum(psyu)
#         ax[i,0].scatter(distxpu,psxu,c=modelc[mi],marker=modelm[mi],zorder=2)
#         ax[i,1].scatter(distypu,psyu,c=modelc[mi],marker=modelm[mi],zorder=2)
        #label axes
        ax[i,0].set_ylabel('{}\n\nMean Probability'.format(ys[i]),fontsize=fs)
        #Set ylims
        ax[i,0].set_ylim([0,.4])
        ax[i,1].set_ylim([0,.4])
        ax[i,1].set_xlabel('y-distance',fontsize=fs)
        if i==1:
            ax[i,0].set_xlabel('x-distance',fontsize=fs)
            ax[i,1].set_xlabel('y-distance',fontsize=fs)
        #Print errors
        ss_errx = sum((psxu-nxp)**2)
        ss_erry = sum((psyu-nyp)**2)
#         print('%s %s error x = %f'%(model.modelshort,ys[i],ss_errx))
#         print('%s %s error y = %f'%(model.modelshort,ys[i],ss_erry))
#         print('%s %s error Total = %f'%(model.modelshort,ys[i],ss_errx+ss_erry))
        prx = pearsonr(psxu,distxpu)
        pry = pearsonr(psyu,distypu)
        print('%s %s corr x = %f, p = %f'%(model.modelshort,ys[i],prx[0],prx[1]))
        print('%s %s corr y = %f, p = %f'%(model.modelshort,ys[i],pry[0],pry[1]))
        

    print('\n')
ax[1,1].set_visible(False)
# ax[0,1].legend({'Packer','Rep'},fontsize=14)      
plt.show()
if saveplots:  
    plt.savefig('private/firstbetas_distcent.pdf',bbox_inches='tight')
#plt.savefig('private/firstbetas_distcentModelOverlay.pdf')