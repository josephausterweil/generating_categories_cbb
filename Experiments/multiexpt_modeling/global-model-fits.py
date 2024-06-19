import pickle
import pandas as pd
import pickle_compat
import os

#this sometimes can find local maxima

pickle_compat.patch()

from io import open

def compile_file(filename):
	with open(filename, encoding='utf-8') as f:
		return compile(f.read(), filename, 'exec')

cur_dir = 'Experiments/multiexpt_modeling'

exec(compile_file(os.path.join(cur_dir,'Imports.py')))


import Modules.Funcs as funcs
from Modules.Classes import Simulation
from Modules.Classes import CopyTweak
from Modules.Classes import Packer
from Modules.Classes import ConjugateJK13
from Modules.Classes import RepresentJK13
import numpy as np

# seed rng
#np.random.seed(9001)

# Specify default dataname
dataname_def = 'pooled'#'pooled-no1st'
participant_def = 'all'
unique_trials_def = 'all'

#Allow for input arguments at the shell
if __name__ == "__main__":
        import sys

        if len(sys.argv)<4:
                unique_trials = unique_trials_def
        else:
                unique_trials = int(sys.argv[3])
        if len(sys.argv)<3:
                participant = participant_def
        else:
                participant = int(sys.argv[2])
        if len(sys.argv)<2:
                dataname = dataname_def
        else:
                dataname = sys.argv[1]        
else:
        dataname = dataname_def
        participant = participant_def
        unique_trials = unique_trials_def

exec(compile_file(os.path.join(cur_dir,'validate_data.py')))

print(f'Fitting Data: ${dataname}')

# get data from pickle
with open(os.path.join(os.path.join(cur_dir,pickledir),src),'rb') as f:
    trials = pickle.load( f ,encoding='latin1')

trials.task = task

print(trials)
trials = Simulation.extractPptData(trials,participant,unique_trials)


# options for the optimization routine
options = dict(
    method = 'Nelder-Mead',
    options = dict(maxiter = 500, disp = False),
    tol = 0.01,
) 

results = dict()
for model_obj in [ConjugateJK13,RepresentJK13,CopyTweak,Packer]:
    res = Simulation.hillclimber(model_obj, trials, options,results=True,callbackstyle='.')
    X = model_obj.params2dict(model_obj.clipper(res.x))
    results[model_obj.model] = X

        #Simulation.show_final_p(model_obj,trials,res.x, show_data = False)
                

for k,v in results.items():
    print(k, v)

        

# save final result in pickle
# with open(pickledir+dst,'wb') as f:
#     pickle.dump(results, f)



