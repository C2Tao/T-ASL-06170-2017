import sys
sys.path.insert(0, '/home/c2tao/zrst')

import cPickle as pickle
import zrst
import zrst.asr
import zrst.util
from zrst import hmm
import os
import numpy as np
from posteriorgram import Posterior
import subprocess
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

def token_dict(path):
    return  os.path.join(path, 'library/dictionary.txt')

def parse_lex(cor, arg_m, arg_n, hier):
    with open(token_dict(token_path(token_name(cor, arg_m, arg_n, hier))),'r') as f:
        #return len(f.readlines())-2
        w = f.readlines()
        u = (max(map(lambda x: len(x.split()),w))-1)*arg_m
        v = len(w)-2
        return u,v 


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
    # dont use, buggy?
    tmp = 'temp'+str(np.random.rand())
    print 'swaping', old, new
    try: os.renames(old, tmp) 
    except: pass
    try: os.renames(new, old) 
    except: pass
    try: os.renames(tmp, new) 
    except: pass
 
def make_hier(cor, arg_m, arg_n):
    tar_path = oken_path(token_name(cor, arg_m, arg_n, 'hier'))
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
    tar_path = token_path(token_name(cor, arg_m, arg_n, 'flat'))
    if os.path.isdir(tar_path):
        print 'lexicon has been flattened at:', tar_path
        return
    A = zrst.asr.ASR(target=tar_path, nState=arg_m)
    A.readASR(token_path(token_name(cor, arg_m, arg_n,'hier')))

    A.x_flatten()

def make_post(cor, arg_m, arg_n, hier):
    tar_path = token_path(token_name(cor, arg_m, arg_n, hier+'post'))
    if os.path.isdir(tar_path):
        print 'lattice has been constructed at:', tar_path
        return
    A = zrst.asr.ASR(target=tar_path, nState=arg_m)
    
    A.readASR(token_path(token_name(cor, arg_m, arg_n, flat_name(hier))))
    if hier=='mult':
         _fix_abs_path(A, cor)

    A.feedback()
    A.lat_testing()

def _parse_timestep(cor):
    wav_list = []
    beg_dict,med_dict,end_dict = {},{},{}
    timestep_path = '/home/c2tao/ZRC_data/{0}/{0}_timesteps.txt'.format(cor)
    with open(timestep_path,'r') as g:
        for line in g:
            #0326 0327 # 415.803 415.819 415.835
            if 'wav'  in line:
                wav = line.split('.')[0]
                wav_list.append(wav)
                beg_dict[wav]=[]
                med_dict[wav]=[]
                end_dict[wav]=[]
            if '#' in line:
                beg_dict[wav].append(line.split()[3])
                med_dict[wav].append(line.split()[4])
                end_dict[wav].append(line.split()[5])
    
    return wav_list, med_dict, beg_dict, end_dict
def _uncut_wav(wav_list):    
    # uncut used only for eng    
    uncut_list = []
    uncut_dict = {}
    for wav in wav_list:
        # uncut used only for eng
        uncut = wav.split('_')[0]
        try:
            uncut_dict[uncut].append(wav)
        except:
            uncut_dict[uncut] = [wav]
            uncut_list.append(uncut)
    return uncut_list, uncut_dict

def make_gram(cor, arg_m, arg_n, hier):
    tar_path = token_path(token_name(cor, arg_m, arg_n, hier+'post'))
    post_path = os.path.join(tar_path, 'posterior')
    if os.path.isdir(post_path):
        print 'posterior has been constructed at:', post_path
        return
    try: os.mkdir(post_path)
    except: pass

    wav_list, med_dict, _, _ = _parse_timestep(cor)
    if cor=='eng':
        uncut_list, uncut_dict = _uncut_wav(wav_list)
    elif cor=='xit':
        uncut_list, uncut_dict = wav_list, dict(zip(wav_list,map(lambda x: [x],wav_list)))
    
    R = Posterior()
    for uncut in uncut_list:
        with open(os.path.join(post_path,uncut+'.pos'),'w') as f:
            for wav in uncut_dict[uncut]:
                lat_path = reduce(os.path.join,[tar_path, 'result', wav+'.lat'])
                if not os.path.isfile(lat_path):
                    #silent files don't have lattices
                    continue
                post = R.read_lat(lat_path)
                #sometimes posterior is 1 frame shorter ?
                meds = np.array(med_dict[wav])[:post.shape[0], None] 

                feat = np.concatenate((meds, post), axis = 1)
                for line in feat:
                    f.write(' '.join(list(line)) + '\n')
        print 'posteriorgram built for:', uncut
    #lat_path2 = '/home/c2tao/mediaeval_token/abx_token/eng_3_300_hierpost/result/s0101a_002.lat'

def make_class(cor, arg_m, arg_n, hier):
    tar_path = token_path(token_name(cor, arg_m, arg_n, hier))
    class_path = os.path.join(tar_path, 'glass.txt')
    mlf_path = os.path.join(tar_path, 'result/result.mlf')
    if os.path.isfile(class_path):
        print 'class file has been constructed at:', class_path
        return
    
    M = zrst.util.MLF(mlf_path)
    _tok = {}
    for i,t in enumerate(M.tok_list): 
        _tok[t]=  i+1

    mwav_list = map(lambda x: x.split('/')[-1],M.wav_list)
    _wav = {}
    for i,w in enumerate(mwav_list):
        _wav[w]= M.wav_list


    wav_list,_, beg_dict, end_dict = _parse_timestep(cor)
    if cor=='eng':
        uncut_list, uncut_dict = _uncut_wav(wav_list)
    elif cor=='xit':
        uncut_list, uncut_dict = wav_list, dict(zip(wav_list,map(lambda x: [x],wav_list)))

    recut_dict = {}
    for u in uncut_dict.keys():
        for w in uncut_dict[u]:
            recut_dict[w] = u
    '''
    Class 1
    s0101a 60.059 60.131
    s0101a 64.709 64.781
    s0101a 79.545 79.617
    s0101a 93.362 93.434
    '''
    tok_time = {}
    print M.tok_list
    for wav, tag, tim in zip(mwav_list, M.tag_list, M.int_list):
        for i, t in enumerate(tag):
            ztim = [0]+tim
            btim = ztim[:-1]
            etim = list(np.array(ztim[1:])-1)
            #print wav,ztim
            #print beg_dict[wav]
            #print end_dict[wav]
            #print len(end_dict[wav])
            try:
                tok_time[t] += (recut_dict[wav], beg_dict[wav][btim[i]], end_dict[wav][etim[i]]),
            except:
                tok_time[t] = [(recut_dict[wav], beg_dict[wav][btim[i]], end_dict[wav][etim[i]])]
    #print tok_time['sil'][:3]
    with open(class_path,'w') as f:
        for t in M.tok_list:
            f.write('Class '+str(_tok[t])+'\n') 
            for tp in tok_time[t]:
                f.write(' '.join(tp)+'\n')
            f.write('\n')


def make_abx(cor, arg_m, arg_n, hier):
    tar_path = token_path(token_name(cor, arg_m, arg_n, hier+'post'))
    post_path = os.path.join(tar_path, 'posterior')
    abx_path = os.path.join(tar_path, 'abx')
    if cor=='eng':
        bin_path ='/home/c2tao/ZRC_revision/eng/eval1/english_eval1/eval1'
    elif cor=='xit':
        bin_path ='/home/c2tao/ZRC_revision/surprise/eval1/xitsonga_eval1'
    subprocess.call([bin_path, '-j','8',post_path, abx_path])
    #~/ZRC_revision/eng/eval1/english_eval1/eval1 -j 8 ../abx_token/eng_3_50_hierpost/posterior eng_eval1

  
def make_seven(cor, arg_m, arg_n, hier):
    tar_path = token_path(token_name(cor, arg_m, arg_n, hier))
    class_path = os.path.join(tar_path, 'glass.txt')
    seven_path = os.path.join(tar_path, 'seven')
    if cor=='eng':
        bin_path ='/home/c2tao/ZRC_revision/eng/eval2/english_eval2'
    elif cor=='xit':
        bin_path ='/home/c2tao/ZRC_revision/surprise/eval2/xitsonga_eval2'
    subprocess.call([bin_path, '-j','8', class_path, seven_path])
                
def make_table(cor, arg_m, arg_n, hier):
    tar_path = token_path(token_name(cor, arg_m, arg_n, hier+'post'))
    abx_path = os.path.join(tar_path, 'abx')
    results_txt = os.path.join(abx_path, 'results.txt')
    #english_within_talkers: 17.626 %
    #english_across_talkers: 27.402 %
    try:
        with open(results_txt,'r') as f:
            lines = f.readlines()
            abx_within = float(lines[1].split(':')[1].strip().split()[0])
            abx_across =  float(lines[2].split(':')[1].strip().split()[0])
            #print '\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(cor, arg_m, arg_n, hier, 'within', abx_within, 'across', abx_across)
            uv = parse_lex(cor, arg_m, arg_n, hier)
            mn = arg_m, arg_n
            if hier =='mult':
                uv = '-'
            print '&{:<12}&{:<12}&{:.2f}  &{:.2f}\\\\'.format(mn, uv, abx_across, abx_within)
            
    except:
        print 'still waiting for results:', tar_path
        

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

def run_make_gram():
    token_args = []
    for c in ['eng','xit']:
        for m in [3, 5, 7]:
            for n in [50, 100, 300, 500]:
                for h in ['hier','mult']:
                    token_args.append((c, m, n, h))
    print token_args
    zrst.util.run_parallel(make_gram, token_args)
    
def run_make_abx():
    for n in [50, 100, 300, 500]:
        for m in [3, 5, 7]:
            for c in ['eng','xit']:
                for h in ['hier','mult']:
                    make_abx(c, m, n, h)

def run_make_seven():     
    token_args = []
    for n in [50, 100]:
        for m in [3, 5, 7]:
            for c in ['eng','xit']:
                for h in ['hier','mult']:
                    token_args.append((c, m, n, h))
    print token_args
    zrst.util.run_parallel(make_class, token_args)
    zrst.util.run_parallel(make_seven, token_args)

def run_make_table():
    for c in ['eng','xit']:
        print c, 'mn,uv,across,within'
        for h in ['mult','hier']:
            if h == 'hier':
                print '\\multirow{6}{*}{Hier.}'
            else:   
                print '\\multirow{6}{*}{Mult.}'
            for m in [3, 5, 7]:
                for n in [50, 100]:#, 300, 500]:
                    make_table(c, m, n, h)
    
if __name__=='__main__':
    #run_make_hier()
    #run_make_flat()
    #run_make_pdist()
    #run_make_post()
    #run_make_gram()
    #run_make_abx()
    #run_make_table()
    #make_class('eng',3,50,'hierpost')
    #make_seven('eng',3,50,'hierpost')
    run_make_seven()
