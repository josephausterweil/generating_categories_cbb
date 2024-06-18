# COMPARES FIT OF CONSTRAINED FORMS OF PACKER TO FULL MODEL (FIG 12 of journal article)
import pickle
import pickle_compat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#originally in Python 2. Converted to 3, so pickles need special attention
pickle_compat.patch()

import os
from io import open
def compile_file(filename):
	with open(filename, encoding='utf-8') as f:
		return compile(f.read(), filename, 'exec')
	
cur_dir = 'Experiments/cogpsych_code/'
exec(compile_file(os.path.join(cur_dir,'Imports.py')))

#execfile('Imports.py')
import Modules.Funcs as funcs
from Modules.Classes import Simulation
from Modules.Classes import Packer, CopyTweak

# get trials pickle
# pickle can be "pickles/all_data_e1_e2.p", "pickles/nosofsky1986.p"
with open(os.path.join(cur_dir,"pickles/all_data_e1_e2.p"), "rb" ) as f:
    trials = pickle.load( f, encoding='latin1' )

# get best params pickle
# pickle can be "pickles/best_params_all_data_e1_e2.p", "pickles/best_params_nosofsky1986.p"
# get best params pickle
with open(os.path.join(cur_dir,"pickles/chtc_gs_best_params_all_data_e1_e2.p"), "rb" ) as f:
    best_params_t = pickle.load( f , encoding='latin1')

#Rebuild it into a smaller dict
best_params = dict()
for modelname in best_params_t.keys():    
    best_params[modelname] = dict()
    for i,parmname in enumerate(best_params_t[modelname]['parmnames']):
        parmval = best_params_t[modelname]['bestparmsll']
        best_params[modelname][parmname] = parmval[i]

# with open("pickles/best_params_all_data_e1_e2.p", "rb" ) as f:
#     best_params = pickle.load( f )

# compute copytweak loglike
start_params = best_params[CopyTweak.model]

# set up grid of param values
gamma_grid = np.linspace(0, 5.0, 100)
loglikes = np.empty(gamma_grid.shape)

# add task type to trials object
trials.task = 'generate'

# evaluate loglike at each grid point
for i, val in enumerate(gamma_grid):
    curr = start_params.copy()
    curr['theta_cntrst'] = val
    curr['theta_target'] = curr['determinism']
    curr = Packer.parmxform(curr, direction = 1)
    loglikes[i] = -trials.loglike(curr, Packer)


for i in range(len(gamma_grid)):
    print(i, gamma_grid[i], loglikes[i])

# plot it
fh = plt.figure(figsize=(5,3))

# equal weight annotation
tempparm = start_params.copy()
tempparm['theta_cntrst'] = tempparm['determinism']
tempparm['theta_target'] = tempparm['determinism']
tempparm = Packer.parmxform(tempparm, direction = 1)
equal_loglike = -trials.loglike(tempparm, Packer)

plt.plot(start_params['determinism'],equal_loglike,'d',linewidth=1,color='k')
plt.text(start_params['determinism']+.2,equal_loglike,r'$\theta_c=\theta_t$')
# copytweak annotation
copytweak_loglike = loglikes[gamma_grid == 0]
plt.plot([min(gamma_grid), max(gamma_grid)], [copytweak_loglike, copytweak_loglike], '--', linewidth = 1, color='gray')
plt.text(2, copytweak_loglike, 'Target Only (Copy & Tweak)', ha = 'center', va = 'bottom',fontsize=12)

# contrast only annotation
contrast_loglike = loglikes[gamma_grid == 2]
#plt.plot([min(gamma_grid), max(gamma_grid)], [contrast_loglike, contrast_loglike], '--', linewidth = 1, color='gray')
#plt.text(0.5, contrast_loglike, 'Contrast Only', ha = 'center', va = 'bottom')

# PACKER annotation
x = float(gamma_grid[loglikes == max(loglikes)])
plt.text(x, max(loglikes) -50, 'PACKER', ha = 'center', va = 'bottom', fontsize = 12)


# plot main line
plt.plot(gamma_grid, loglikes,'k-', linewidth = 1)
plt.xlabel(r'$\theta_c$ Parameter Value', fontsize = 12)

xstep = 1.0
xticks = np.round(np.arange(min(gamma_grid),max(gamma_grid)+xstep,xstep),1)
xticklabels = [str(i) for i in xticks]
xticklabels[0] = '0'
#xticklabels[-1] = '1'
plt.xticks(xticks,fontsize=12)
#fh.gca().set_xticklabels(xticklabels)

yticks = np.arange(-4950,-4650,50)
yticklabels = [str(i) for i in yticks]
plt.yticks(yticks,fontsize=12)
fh.gca().set_yticklabels(yticklabels)

plt.gca().yaxis.grid(True)
plt.ylabel('Log-Likelihood ($L$)', fontsize = 12)

plt.savefig('packer-loglike.pdf', bbox_inches='tight', transparent=False)
# path = '../../Manuscripts/cog-psych/revision/figs/packer-loglike.pgf'
# funcs.save_as_pgf(fh, path)
