import sys
sys.path.insert(0, '/home/c2tao/zrst')

import cPickle as pickle
import zrst
import zrst.asr
import zrst.util
from zrst import hmm
import os
import numpy as np
#def wav(cor):
#    if cor =='eng': return '/home/c2tao/ZRC_revision/eng/preprocess/short_wavs'
#    elif cor =='xit': return '/home/c2tao/ZRC_revision/surprise/preprocess/cut_wavs'
#def mfc_name(cor):
#    if cor =='eng': return '/home/c2tao/ZRC_revision/eng/preprocess/mfc'
#    elif cor=='xit': return '/home/c2tao/ZRC_revision/surprise/preprocess/cut_mfc'


def cor_name(cor):
    return 'eng' if cor=='eng' else 'surprise'

def flat_name(hier):
    return 'flat' if hier=='hier' else hier

def wav_path(cor):
    return '/home/c2tao/mediaeval_token/abx_corpus/{}'.format(cor)

def token_name(cor, m, n, hier):
    return "_".join([cor, str(m), str(n), hier])

def token_pdist(tok_name):
    return '/home/c2tao/mediaeval_token/abx_pdist/{}.dat'.format(tok_name)

def token_path(tok_name):
    #'~/ZRC_revision/eng/zrstExps/mfc_50_3'
    #50,100,300,500
    #3,5,7

    #'~/ZRC_revision/surprise/zrstExps/mfc_50_3'
    #50,100,300,500
    #3,5,7,9
    cor, m, n, hier = tok_name.split("_")
    if hier == 'mult':
        if cor =='eng':
            return '/home/c2tao/ZRC_revision/eng/zrstExps/mfc_{}_{}/'.format(str(n), str(m))
        elif cor=='xit':
            return '/home/c2tao/ZRC_revision/surprise/zrstExps/mfc_{}_{}/'.format(str(n), str(m))
    else:
        return '/home/c2tao/mediaeval_token/abx_token/{}/'.format(tok_name)


def _fix_abs_path(A, cor):
    #change to absolute path
    old_file = A.X['wavlst_scp']
    new_file = old_file+'_alt'
    with open(old_file,'r') as f, open(new_file,'w') as g:
        for line in f:
            g.write('/home/c2tao/ZRC_revision/'+cor_name(cor)+'/zrstExps/'+line)

    #swap file name
    t = 'temp.file'
    os.rename(new_file, t)
    os.rename(old_file, new_file)
    os.rename(t, old_file)
    

def _swap(old, new):
    tmp = 'temp'+str(np.random.rand())
    print 'swaping', old, new
    try: os.renames(old, tmp) 
    except: pass
    try: os.renames(new, old) 
    except: pass
    try: os.renames(tmp, new) 
    except: pass
 
def make_hier(cor, arg_m, arg_n):
    tar_path = target=token_path(token_name(cor, arg_m, arg_n, 'hier'))
    if os.path.isdir(tar_path):
        print 'training finished:', tar_path
        return
    A = zrst.asr.ASR(target=tar_path, nState=arg_m)
    A.readASR(token_path(token_name(cor, arg_m, arg_n,'mult')))
    _fix_abs_path(A, cor)

    A.x(3)
    A.a_keep()
    A.a_keep()

def make_flat(cor, arg_m, arg_n):
    tar_path = target=token_path(token_name(cor, arg_m, arg_n, 'flat'))
    if os.path.isdir(tar_path):
        print 'lexicon has been flattened at:', tar_path
        return
    A = zrst.asr.ASR(target=tar_path, nState=arg_m)
    A.readASR(token_path(token_name(cor, arg_m, arg_n,'hier')))

    A.x_flatten()

def make_post(cor, arg_m, arg_n, hier):
    tar_path = target=token_path(token_name(cor, arg_m, arg_n, hier+'post'))
    if os.path.isdir(tar_path):
        print 'lattice has been constructed at:', tar_path
        return
    A = zrst.asr.ASR(target=tar_path, nState=arg_m)
    
    A.readASR(token_path(token_name(cor, arg_m, arg_n, flat_name(hier))))
    if hier=='mult':
         _fix_abs_path(A, cor)

    A.feedback()
    A.lat_testing()
    

def make_pdist(cor, arg_m, arg_n, hier):
    tok_name = token_name(cor, arg_m, arg_n, hier)
    try:
        dm = pickle.load(open(token_pdist(tok_name),'r'))
        print 'loaded precomputed distance for ' + tok_name
        return dm
    except:
        print 'computing distance for ' + tok_name
    dm = {}
    H = hmm.parse_hmm(token_path(tok_name)+'hmm/models')
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
    pickle.dump(dm,open(token_pdist(tok_name),'w'))
    return dm


def run_make_hier():
    token_args = []
    for c in ['eng','xit']:
        for m in [3, 5, 7]:
            for n in [50, 100, 300, 500]:
                token_args.append((c, m, n))
    print token_args
    zrst.util.run_parallel(make_hier, token_args)

def run_make_flat():
    token_args = []
    for c in ['eng','xit']:
        for m in [3, 5, 7]:
            for n in [50, 100, 300, 500]:
                token_args.append((c, m, n))
    print token_args
    zrst.util.run_parallel(make_hier_flat, token_args)

def run_make_post():
    token_args = []
    for c in ['eng','xit']:
        for m in [3, 5, 7]:
            for n in [50, 100, 300, 500]:
                for h in ['hier','mult']:
                    token_args.append((c, m, n, h))
    print token_args
    zrst.util.run_parallel(make_post, token_args)

def run_make_pdist():
    token_args = []
    for c in ['eng','xit']:
        for m in [3, 5, 7]:
            for n in [50, 100, 300, 500]:
                for h in ['hier','mult']:
                    token_args.append((c, m, n, h))
    print token_args
    zrst.util.run_parallel(make_pdist, token_args)


if __name__=='__main__':
    #make_hier('eng',7, 50)
    #run_make_hier()
    #run_make_flat()
    #run_make_pdist()
    run_make_post()
