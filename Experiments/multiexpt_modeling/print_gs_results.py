#from Modules.Classes import Simulation
import re
import os
import pickle
import pickle_compat

pickle_compat.patch()

def compile_file(filename):
	with open(filename, encoding='utf-8') as f:
		return compile(f.read(), filename, 'exec')

cur_dir = 'Experiments/multiexpt_modeling'

exec(compile_file(os.path.join(cur_dir,'Imports.py')))


#Find all gs fits and print them. Nice nice.        
pickledir = os.path.join(cur_dir,'pickles/')
prefix = 'chtc_gs_best_params'
#Compile regexp obj
allfiles =  os.listdir(pickledir)
r = re.compile(prefix)
gsfiles = filter(r.match,allfiles)

#Compute total SSE


for i,file in enumerate(gsfiles):
    #Extract key data from each file
    print('\n' + file)
    print('------')
    with open(pickledir+file,'rb') as f:
        fulldata = pickle.load(f,encoding='latin1')
    modelnames = fulldata.keys()
    for j in modelnames:
        print(j)
        for pi,pname in enumerate(fulldata[j]['parmnames']):
            print('\t' + pname + ': ' + str(fulldata[j]['bestparmsll'][pi]))
        print('\tLogLike' + ' = ' + '-' + str(fulldata[j]['bestparmsll'][pi+1]))
        print('\tAIC'  + ' = ' + str(fulldata[j]['bestparmsll'][pi+2]) + '\n')
        # for pi,pname in enumerate(fulldata[j]['parmnames']):
        #     print str(fulldata[j]['bestparmsll'][pi])
            
        # print  str(fulldata[j]['bestparmsll'][pi+1]) 
        # print  str(fulldata[j]['bestparmsll'][pi+2]) + '\n'

