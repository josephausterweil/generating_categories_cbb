import pickle
import pickle_compat

pickle_compat.patch()

import pandas as pd
import os

def compile_file(filename):
	with open(filename, encoding='utf-8') as f:
		return compile(f.read(), filename, 'exec')

cur_dir = 'Experiments/corner/modeling'

exec(compile_file(os.path.join(cur_dir,'Imports.py')))

import Modules.Funcs as funcs
from Modules.Classes import Simulation
from Modules.Classes import CopyTweak
from Modules.Classes import Packer
from Modules.Classes import ConjugateJK13


# all data
src = "pickles/all_trials.p"
dst = "pickles/best_params_all_trials.p"

# # trials 2-4
src = "pickles/trials_2-4.p"
dst = "pickles/best_params_trials.p"

# get data from pickle
with open(os.path.join(cur_dir,src), "rb" ) as f:
	trials = pickle.load( f , encoding='latin1')

# options for the optimization routine
options = dict(
	method = 'Nelder-Mead',
	options = dict(maxiter = 500, disp = False),
	tol = 0.01,
) 

results = dict()
for model_obj in [ConjugateJK13, CopyTweak, Packer]:

	res = Simulation.hillclimber(model_obj, trials, options)
	X = model_obj.params2dict(model_obj.clipper(res.x))
	results[model_obj.model] = X

# save final result in pickle
# with open(dst,'wb') as f:
# 	pickle.dump(results, f)

for k,v in results.items():
	print(k, v)