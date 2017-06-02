
import zrst
import zrst.util
import zrst.asr
from zrst import hmm
import cPickle as pickle
import numpy as np

path_root = '/home/c2tao/'
path_matlab = path_root + 'zrst/zrst/matlab/'

wav = {}
wav_root = path_root + 'mediaeval_2015/data/'
wav['doc'] = wav_root + 'QUESST2015-dev/Audio/'
wav['dev'] = wav_root + 'QUESST2015-dev/dev_queries/' 
wav['eva'] = wav_root + 'QUESST2015-eval_groundtruth/eval_queries/'

#init = {}
#init_root = path_root + 'mediaeval_token/init/'
#init['doc'] = lambda n: init_root + 'doc_{}_flatten.txt'.format(n)
#init['dev'] = lambda n: init_root + 'dev_{}_flatten.txt'.format(n)
#init['eva'] = lambda n: init_root + 'eva_{}_flatten.txt'.format(n)


init_name = lambda c, n: '{}_{}'.format(c, n)
init = lambda name: path_root + 'mediaeval_token/init/{}_flatten.txt'.format(name)

token_name = lambda c, m, n: '{}_{}_{}'.format(c, m, n)
token = lambda name: path_root + 'mediaeval_token/token/{}/'.format(name)
pdist = lambda name: path_root + 'mediaeval_token/pdist/{}.dat'.format(name)



'''
def make_init(in_wav, arg_cluster, out_init):
    import subprocess 
    subprocess.Popen(['./run_clusterDetection.sh', in_wav, str(arg_cluster), out_init], cwd = path_matlab)
    #subprocess.Popen('./run_clusterDetection.sh ~/mediaeval_2015/data/QUESST2015-dev/Audio/ 500 mediaeval_500.txt', cwd = path_matlab)

def run_make_init():
    for n in [100, 200, 300]:
        make_init(wav_dev, n, init_dev.format(n))
        make_init(wav_eva, n, init_eva.format(n))
'''

def make_init():
    import os
    for f in os.listdir(init_root):
        if f[-4:]=='.txt':
            _in = init_root+f
            _out = init_root+f[:-4]+'_flatten'+f[-4:]
            print _in, _out
            zrst.util.flat_dumpfile(_in, _out)
    
def make_token(cor, arg_m, arg_n):
    A = zrst.asr.ASR(corpus=wav[cor], target=token(token_name(cor, arg_m, arg_n)), dump=init(init_name(cor,arg_n)), do_copy=True, nState=arg_m)
    A.initialization()
    A.iteration('a')
    A.iteration('a_keep')
    A.iteration('a_keep')
    A.iteration('a_keep')
    A.iteration('a_keep')
    '''
    for i in range(3):
        A.iteration('a_mix', config_name)
        for i in range(10):
            A.iteration('a_keep')
    '''

def make_pdist(cor, arg_m, arg_n):
    tokset_name = token_name(cor, arg_m, arg_n)
    try:
        dm = pickle.load(pdist(tokset_name),'r')
        print 'loaded precomputed distance for ' + tokset_name
        return dm
    except:
        print 'computing distance for ' + tokset_name
    dm = {}
    H = hmm.parse_hmm(token(tokset_name)+'hmm/models')
    phones = filter(lambda x: 's' not in x,H.keys())
    for i in phones:
        for j in phones:
            dm[(i,j)] =  hmm.kld_hmm(H[i],H[j])

    ok = set([])
    for k in dm.keys():
        ok.add(k[0])
        ok.add(k[1])
    ok = list(ok)

    for k in ok:
        ambig = np.mean([dm[x] for x in dm.keys() \
                         if (x[0] == k or x[1] == k) and dm[x] != float('Inf')])
        dm[('sil', k)] = ambig
        dm[('sp', k)] = ambig
    ambig = np.mean([dm[x] for x in dm.keys() \
                     if dm[x] != float('Inf')])
    dm[('sil', 'sp')] = ambig
    dm[('sil', 'sil')] = ambig
    dm[('sp', 'sp')] = ambig

    for k in dm.keys():
        dm[(k[1], k[0])] = dm[k]
    pickle.dump(dm,open(pdist(tokset_name),'w'))
    return dm

def run_make_pdist():
    import sys
    token_args = []
    for c in ['dev','eva']:
        for m in [3, 5, 7]:
            for n in [10, 100, 200, 300]:
                token_args.append((c, m, n))
    for c in ['doc']:
        for m in [3, 5, 7]:
            for n in [100, 200]:
                token_args.append((c, m, n))
    print token_args
    zrst.util.run_parallel(make_pdist, token_args)
    ##print map(token_name, *zip(*token_args))
    #for t in tok_args:
    #    if sys.argv[1] not in t: continue
    #    X = make_pdist(t)
    #    print 'progress', 100.0*tokset.index(t)/len(tokset),'%'


    
def ran_qer_token():
    token_args = []
    for c in ['dev','eva']:
        for m in [3, 5, 7]:
            for n in [10, 100, 200, 300]:
                token_args.append((c, m, n))
    print token_args
    zrst.util.run_parallel(make_token, token_args)

def ran_doc_token():
    token_args = []
    for c in ['doc']:
        for m in [3, 5, 7]:
            for n in [100, 200]:
                token_args.append((c, m, n))
    print token_args
    zrst.util.run_parallel(make_token, token_args)


if __name__=='__main__':
    #make_init()
    #ran_qer_token()
    #ran_doc_token()
    run_make_pdist() 
     
