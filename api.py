import sys
sys.path.insert(0, '/home/c2tao/zrst')

import os
import zrst
import zrst.util
#import zrst.asr
from zrst import hmm
import cPickle as pickle
import numpy as np
from score import parse_eval, write_mediaeval_score
import dtw
import subprocess 
from subprocess import PIPE
import time

path_root = '/home/c2tao/'
answer_root = '/home/c2tao/mediaeval_2015/data/scoring_quesst2015_lang_noise/'
path_matlab = path_root + 'zrst/zrst/matlab/'

wav = {}
wav_root = path_root + 'mediaeval_2015/data/'
wav['doc'] = wav_root + 'QUESST2015-dev/Audio/'
wav['dev'] = wav_root + 'QUESST2015-dev/dev_queries/' 
wav['eva'] = wav_root + 'QUESST2015-eval_groundtruth/eval_queries/'

init_name = lambda cor, n: '{}_{}'.format(cor, n)
init = lambda name: path_root + 'mediaeval_token/init/{}_flatten.txt'.format(name)

#token_name = lambda cor, m, n: '{}_{}_{}'.format(cor, m, n)
token_name = lambda cor, m, n, hier='mult': '{}_{}_{}{}'.format(cor, m, n,'_hier' if hier=='hier' else '')
token = lambda name: path_root + 'mediaeval_token/token/{}/'.format(name)
token_fix = lambda name: path_root + 'mediaeval_token/token/{}_snapshot/6_{}/'.format(name,name)
token_pdist = lambda name: path_root + 'mediaeval_token/pdist/{}.dat'.format(name)
token_result = lambda name: path_root + 'mediaeval_token/token/{}/result/result.mlf'.format(name)

hier_old = lambda name: path_root + 'mediaeval_token/hier/{}/'.format(name)
hier_lex = lambda name: path_root + 'mediaeval_token/hier/{}_lex/'.format(name)
hier_dict = lambda name: path_root + 'mediaeval_token/hier/{}_lex/library/dictionary.txt'.format(name)


decode_name = lambda tok, cor: '{}_{}'.format(tok, cor)
decode = lambda name: path_root + 'mediaeval_token/decode/{}.mlf'.format(name)

cor_mfcc = lambda cor: path_root + 'mediaeval_token/decode/list_{}.scp'.format(cor)


method_name = ['whole','short','warp']

score_name = lambda qer, tok, method: '{}_{}_{}'.format(qer, tok, method)
score = lambda name: path_root+ 'mediaeval_token/score/{}.dat'.format(name)

evals_name = lambda scr, ans: '{}{}'.format(scr, '_'+ans if ans!='' else '' )
evals = lambda name: path_root + 'mediaeval_token/eval/{}/score.out'.format(name)
evals_file = lambda name: path_root + 'mediaeval_token/eval/{0}/{0}.stdlist.xml'.format(name)
evals_dir = lambda name: path_root + 'mediaeval_token/eval/{}/'.format(name)



#answ_name=\
'''              
_Albanian     _Chinese      _Czech        _Portuguese   _Romanian     _Slovak  
_rvb0_db1     _rvb0_db14    _rvb0_db2     _rvb0_db3     _rvb0_db4     _rvb0_db5     _rvb0_db5_T1  
_rvb1_db1     _rvb1_db2     _rvb1_db3     _rvb1_db4     _rvb1_db5     
_rvb2_db1     _rvb2_db2  _rvb2_db3  _rvb2_db4  _rvb2_db5  
_rvb3_db1  _rvb3_db2  _rvb3_db3  _rvb3_db4  _rvb3_db5  
_rvb4_db1  _rvb4_db2  _rvb4_db3  _rvb4_db4  _rvb4_db5  
_rvb5_db1  _rvb5_db2  _rvb5_db3  _rvb5_db4  _rvb5_db5
_rvb15_db14   _rvb15_db5    
_T1      _T2      _T3      
'''
answer = lambda qer, ans: answer_root + 'groundtruth_quesst2015_{}{}'.format('dev' if qer=='dev' else 'eval', ans if ans=='' else '_'+ans)

dev_list = ['quesst2015_dev_{0:04}'.format(q+1) for q in range(445)]
eva_list = ['quesst2015_eval_{0:04}'.format(q+1) for q in range(447)]
doc_list = ['quesst2015_{0:05}'.format(q+1) for q in range(11662)]

def fix_token(c,m,n):
    wrongdir = token(token_name(c,m,n))
    rightdir = token_fix(token_name(c,m,n))
    #subprocess.Popen(['rm', '-rf', wrongdir])
    #subprocess.Popen(['cp', '-r', rightdir, wrongdir])
    subprocess.Popen(['mv', hier_old(token_name(c,m,n)), token(token_name(c,m,n,'hier'))])

def make_list(wav_folder):
    os.listdir(wav_list)
    cor_mfcc = lambda wav_name: ['\"{}\"'.format(f) for f in sorted(os.listdir(wav_name[:-1]+'_MFCC/'))]#quesst2015_dev_0290.mfc
    
    

def make_init():
    for f in os.listdir(init_root):
        if f[-4:]=='.txt' and 'flatten' not in f:
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
def make_hier(cor, arg_m, arg_n):
    A = zrst.asr.ASR(corpus=wav[cor], target=hier_old(token_name(cor, arg_m, arg_n, 'mult')), dump=init(init_name(cor,arg_n)), do_copy=True, nState=arg_m)
    A.readASR(token(token_name(cor, arg_m, arg_n)))
    A.x(3)
    A.a_keep()
    A.a_keep()
    A.x_flatten()
    
def make_hier_lex(cor, arg_m, arg_n):
    A = zrst.asr.ASR(corpus=wav[cor], target=hier_lex(token_name(cor, arg_m, arg_n, 'mult')), dump=init(init_name(cor,arg_n)), do_copy=True, nState=arg_m)
    A.readASR(hier_old(token_name(cor, arg_m, arg_n, 'mult')))
    A.x(3)

def parse_lex(cor, arg_m, arg_n):
    with open(hier_dict(token_name(cor, arg_m, arg_n)),'r') as f:
        #return len(f.readlines())-2
        w = f.readlines()
        u = (max(map(lambda x: len(x.split()),w))-1)*arg_m
        v = len(w)-2
        return u,v 
def make_pdist(cor, arg_m, arg_n, hier='mult'):
    tok_name = token_name(cor, arg_m, arg_n, hier)
    try:
        dm = pickle.load(open(token_pdist(tok_name),'r'))
        print 'loaded precomputed distance for ' + tok_name
        return dm
    except:
        print 'computing distance for ' + tok_name
    dm = {}
    H = hmm.parse_hmm(token(tok_name)+'hmm/models')
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

    
def make_decode(hmm_cor, arg_m, arg_n, dec_cor, hier='mult'):
    tok_name = token_name(hmm_cor, arg_m, arg_n, hier)
    dec_name = decode_name(tok_name, dec_cor)
    A = zrst.asr.ASR(target=token(tok_name))
    A.external_testing(cor_mfcc(dec_cor), decode(dec_name))

def calc_dist(qtag, dtag, pdmat, method, hmm_prob=None):
    dmat = np.zeros([len(qtag),len(dtag)])
    
    for i, d in enumerate(dtag):
        for j, q in enumerate(qtag):
            dmat[j,i] = pdmat[(d,q)]
    #print len(qtag),qtag
    #print len(dtag),dtag
    if method =='whole':
        if len(dtag)-len(qtag)<0: return np.median(dmat)
        vdis = np.zeros(len(dtag)-len(qtag)+1)
        for i in range(len(dtag)-len(qtag)+1):
            v = np.diag(dmat,i)
            vdis[i] = np.sum(v)/len(v)
            #print np.sum(v)/len(v)
        return -np.mean(np.sort(vdis)[:3])
    elif method == 'short':
        vdis = np.zeros(len(qtag)+len(dtag)-1)
        for i in np.arange(-len(qtag)+1,len(dtag)):
            v = np.diag(dmat,i)
            vdis[i+len(qtag)-1] = np.sum(v)/len(v)
            #print np.sum(v)/len(v)
        return -np.mean(np.sort(vdis)[:3])
    elif method == 'norm':
        dmat = dmat/np.linalg.norm(dmat)
        vdis = np.zeros(len(qtag)+len(dtag)-1)
        for i in np.arange(-len(qtag)+1,len(dtag)):
            v = np.diag(dmat,i)
            vdis[i+len(qtag)-1] = np.sum(v)/len(v)
            #print np.sum(v)/len(v)
        return -np.mean(np.sort(vdis)[:3])
    elif method =='hmmp':
        dmat = dmat/np.linalg.norm(dmat)
        dmat = dmat+hmm_prob
        vdis = np.zeros(len(qtag)+len(dtag)-1)
        for i in np.arange(-len(qtag)+1,len(dtag)):
            v = np.diag(dmat,i)
            vdis[i+len(qtag)-1] = np.sum(v)/len(v)
        return -np.mean(np.sort(vdis)[:3])
    elif method =='only':
        dmat = hmm_prob
        vdis = np.zeros(len(qtag)+len(dtag)-1)
        for i in np.arange(-len(qtag)+1,len(dtag)):
            v = np.diag(dmat,i)
            vdis[i+len(qtag)-1] = np.sum(v)/len(v)
        return -np.mean(np.sort(vdis)[:3])
        
        
    elif method =='debug':
        print dmat.shape
        qprob,dprob = hmm_prob
        dmat = 10.0*dmat/np.linalg.norm(dmat)
        dprob = 1.0* -dprob/np.linalg.norm(dprob)
        qprob = 1.0*-qprob/np.linalg.norm(qprob)

        if len(dtag)-len(qtag)<0: 
            vdis = 1.0*np.ones(3)*np.median(dmat)
        else:
            vdis = np.zeros(len(dtag)-len(qtag)+1)
            for i in range(len(dtag)-len(qtag)+1):
                v = np.diag(dmat,i)
                vdis[i] = np.sum(v)/len(v)

        dmat = (dmat.T + qprob).T
        dmat = dmat + dprob
        vdis2 = np.zeros(len(qtag)+len(dtag)-1)
        for i in np.arange(-len(qtag)+1,len(dtag)):
            v = np.diag(dmat,i)
            vdis2[i+len(qtag)-1] = np.sum(v)/len(v)
        
        return np.mean(np.sort(vdis2)[:3])
        #return np.mean(np.sort(vdis)[:3])
        #return np.min(vdis2)
        #return np.min(vdis)



def calc_score(qer_path, doc_path, pdmat, method):
    dmlf = zrst.util.MLF(doc_path)
    qmlf = zrst.util.MLF(qer_path)
    if method =='debug':
        #conclusions so far: 
        #1)hmm dist alone is better
        #2)dtw is not only usless, it is harmful
        
        def dsort(x):
            zz = sorted(zip(x,range(len(x))), key = lambda x:x[0])
            a,b = zip(*zz)
            return b
        qql = dsort(qmlf.wav_list)
        ddl = dsort(dmlf.wav_list)
        def qq(index):
            return qql[index-1]
        def dd(index):
            return ddl[index-1]
        ss = []
        print dsort(['s3','s1','s4','s2'])
        #for j, qtag in enumerate(qmlf.tag_list):
        #    for i, dtag in enumerate(dmlf.tag_list)
        #print 'pdmat',pdmat[('p1','p1')]
        qf,df = 31,126
        #qf,df = 390,46
        glist = [(qq(qf),dd(df))] + map(lambda t: (qq(qf), dd(t+1)), range(11662)) + map(lambda t: (qq(t+1), dd(df)), range(445))
        
        '''
        for j,i in [(qq(31),dd(126)),\
                    (qq(31),dd(127)),\
                    (qq(31),dd(128)),\
                    (qq(31),dd(129)),\
                    (qq(32),dd(126)),\
                    (qq(33),dd(126)),\
                    (qq(34),dd(126))]:
        '''
        for j,i in glist:
            # True 
            #if not( qmlf.wav_list[j]=='quesst2015_dev_0390' and dmlf.wav_list[i]=='quesst2015_00046'): continue # 
            #if not( qmlf.wav_list[j]=='quesst2015_dev_0031' and dmlf.wav_list[i]=='quesst2015_00126'): continue

            qtag, dtag = qmlf.tag_list[j],dmlf.tag_list[i]
            #print dtag
            #print qtag
            q = int(qmlf.wav_list[j].split('_')[-1])
            d = int(dmlf.wav_list[i].split('_')[-1])
            #print q,d
            qprob,dprob =  (np.array(qmlf.log_list[j]), np.array(dmlf.log_list[i]))
            #print qprob
            #print dprob
            #print token_pdist[(dtag[0],qtag[0])]
            #scores[q,d] = zrst.util.SubDTW(dtag, qtag, lambda a,b: token_pdist[(a,b)])
            vtag = calc_dist(qtag[1:-1], dtag[1:-1], pdmat, method, (qprob[1:-1],dprob[1:-1]))
            ss.append(vtag)
            #print vtag
        xx = ss[1:]-ss[0] 
        print 'larger better',np.sum(xx)
        #print dmlf.wav_list[0], qmlf.wav_list[0]
    scores = np.zeros([len(qmlf.wav_list), len(dmlf.wav_list)])
    if method=='zero':
        return scores
    if method=='rand':
        return np.random.rand(len(qmlf.wav_list), len(dmlf.wav_list))
    if method =='hmmp':
        for j, qtag in enumerate(qmlf.tag_list):
            print qer_path, 100.0*j/len(qmlf.tag_list)
            qprob =  np.array(qmlf.log_list[j][1:-1])
            qprob = -qprob/np.linalg.norm(qprob)/10.0
            qprob = qprob[:,np.newaxis]
            for i, dtag in enumerate(dmlf.tag_list):
                d = int(dmlf.wav_list[i].split('_')[-1])-1
                q = int(qmlf.wav_list[j].split('_')[-1])-1
                dprob =  np.array(dmlf.log_list[i][1:-1])
                dprob = -dprob/np.linalg.norm(dprob)/10.0
                dprob = dprob[np.newaxis,:]
                scores[q,d] = calc_dist(qtag[1:-1], dtag[1:-1], pdmat, method, qprob+dprob)
    else: 
        for j, qtag in enumerate(qmlf.tag_list):
            print qer_path, 100.0*j/len(qmlf.tag_list)
            for i, dtag in enumerate(dmlf.tag_list):
                d = int(dmlf.wav_list[i].split('_')[-1])-1
                q = int(qmlf.wav_list[j].split('_')[-1])-1
                if method =='warp':
                    scores[q,d] = dtw.pattern_dtw(qtag[1:-1], dtag[1:-1], pdmat)
                elif method=='whole' or 'short':
                    scores[q,d] = calc_dist(qtag[1:-1], dtag[1:-1], pdmat, method)
                
    return scores




def make_score(qer, cor, m, n, method, hier='mult'):
    #qer, cor, m, n, method = score_name.split('_')
    tok_name = token_name(cor, m, n, hier) 
    scr_name = score_name(qer, tok_name, method)
    try:
        sc = pickle.load(open(score(scr_name),'r'))
        print 'loaded precomputed score for ' + scr_name
        return sc
    except:
        print 'computing score for ' + scr_name

    if cor=='doc':
        doc_mlf = token_result(tok_name)
        qer_mlf = decode(decode_name(tok_name, qer))
    else:
        print 'cor,qer',cor, qer
        if cor==qer:
            qer_mlf = token_result(tok_name)
        else:
            qer_mlf = decode(tok_name, qer)
        doc_mlf = decode(decode_name(tok_name, 'doc'))

    sc = calc_score(qer_mlf, doc_mlf, make_pdist(cor, m, n, hier), method)
    if method !='debug':
        with open(score(scr_name),'w') as f:
            pickle.dump(sc, f, protocol=pickle.HIGHEST_PROTOCOL)
        return sc


def make_eval(qer, cor, m, n, method, ans, hier = 'mult',thresh=None):
    scr_name = score_name(qer, token_name(cor, m, n, hier), method)
    eva_name = evals_name(scr_name, ans)
    try:
        evaluation = parse_eval(evals(eva_name))
        #print 'loaded precomputed evaluation for ' + eva_name+ ' {0:0.4f}'.format(evaluation[-1])
        #print 'loaded precomputed evaluation for ' + eva_name+ ' {0:0.4f}'.format(evaluation[3])
        #print 'loaded precomputed evaluation for ' + eva_name+ ' {0:0.4f}'.format(evaluation[4])
        #eval 9 
        #print 'loaded precomputed evaluation for ' + eva_name+ ' {0:0.4f}'.format(evaluation[9])
        if hier=='hier':
            u, v =  parse_lex(cor,m,n)
        else:
            u, v = '',''
        return method, cor, (m, n), (u, v), evaluation[9]
    except:
        #print 'computing evaluation for ' + eva_name
        pass

    if not os.path.isfile(score(scr_name)):
        #print 'no score calculated for', scr_name
        return
    evaldir = evals_dir(eva_name) 
    evalfile = evals_file(eva_name)
    subprocess.Popen(['rm', '-rf', evaldir])
    subprocess.Popen(['mkdir', evaldir])
    sc = make_score(qer, cor, m, n, method, hier)
    if thresh==None:
        thresh = np.median(sc)
    if qer=='dev':
        qer_list = dev_list
    else:
        qer_list = eva_list

    yesno = sc < thresh
    write_mediaeval_score(eva_name, '', yesno,  sc, qer_list, doc_list, evalfile)
    #subprocess.Popen(['cp', evals_file(score_name), evals_dir(score_name)])
    subprocess.Popen(['./score-TWV-Cnxe.sh', evaldir, answer(qer, ans)], cwd = answer_root)
    time.sleep(50)
    '''
    output, error = p.communicate()
    if p.returncode!=0:
        print '#################'
        print output, error
    '''
    subprocess.Popen(['rm', evalfile])
    #DET.pdf  score.out  TWV.pdf
    subprocess.Popen(['rm', 'DET.pdf'], cwd = evaldir)
    subprocess.Popen(['rm', 'TWV.pdf'], cwd = evaldir)
    evaluation = parse_eval(evals(eva_name))
    print evaluation
    print '{0:0.4f}'.format(evaluation[-1])
    return evaluation

 
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

def run_qer_token():
    token_args = []
    for c in ['dev','eva']:
        for m in [3, 5, 7]:
            for n in [10, 100, 200, 300]:
                token_args.append((c, m, n))
    print token_args
    zrst.util.run_parallel(make_hier, token_args)
    zrst.util.run_parallel(make_hier_lex, token_args)
    zrst.util.run_parallel(fix_token, token_args)

def run_doc_token():
    token_args = []
    for c in ['doc']:
        for m in [3, 5, 7]:
            for n in [100, 200]:
                token_args.append((c, m, n))
    print token_args
    zrst.util.run_parallel(make_hier, token_args)
    zrst.util.run_parallel(make_hier_lex, token_args)
    zrst.util.run_parallel(fix_token, token_args)

def ran_make_pdist(hier='mult'):
    token_args = []
    '''
    for c in ['dev','eva']:
        for m in [3, 5, 7]:
            for n in [10, 100, 200, 300]:
                token_args.append((c, m, n, hier))
    '''
    for c in ['doc']:
        for m in [3, 5, 7]:
            for n in [100, 200]:
                token_args.append((c, m, n, hier))
    print token_args
    zrst.util.run_parallel(make_pdist, token_args)

def ran_make_decode(hier='mult'):
    token_args = []
    '''
    for hmm_c in ['dev','eva']:
        for m in [3, 5, 7]:
            for n in [10, 100, 200, 300]:
                for dec_c in ['doc']:
                    token_args.append((hmm_c, m, n, dec_c, hier))
    '''
    for hmm_c in ['doc']:
        for m in [3, 5, 7]:
            for n in [100, 200]:
                for dec_c in ['dev','eva']:
                    token_args.append((hmm_c, m, n, dec_c, hier))
    print token_args
    zrst.util.run_parallel(make_decode, token_args)

def run_make_score(hier='mult'):
    token_args = []
    '''
    for qer in ['dev','eva']:
        for hmm_c in ['dev','eva','doc']:
            for m in [3,5,7]:
                for n in [100, 200]:
                    for method in ['whole','short', 'warp']:
                        token_args.append((qer,hmm_c,m,n,method))
    for qer in ['dev','eva']:
        for hmm_c in [qer, 'doc']:
            for m in [3,5,7]:
                for n in [100,200]:
                    for method in ['hmmp','short', 'norm','only']:
                        token_args.append((qer,hmm_c,m,n,method, hier))
    for qer in ['dev']:
        for hmm_c in ['doc']:
            for m in [3,5,7]:
                for n in [100,200]:
                    for method in ['hmmp','short', 'norm']:
                        token_args.append((qer,hmm_c,m,n,method, hier))
    for qer in ['dev']:
        for hmm_c in ['dev']:
            for method in ['hmmp','norm']:
                for m in [3,5,7]:
                    for n in [100,200]:
                        token_args.append((qer,hmm_c,m,n,method, hier))
    '''
    for qer in ['dev']:
        for hmm_c in ['doc']:
            for method in ['hmmp','norm']:
                for m in [3,5,7]:
                    for n in [100,200]:
                        token_args.append((qer,hmm_c,m,n,method, hier))
    zrst.util.run_parallel(make_score, token_args)

    

def run_make_eval(hier='mult'):
    '''
    for qer in ['dev','eva']:
        for hmm_c in ['dev','eva','doc']:
            for m in [3,5,7]:
                for n in [100, 200]:
                    for method in ['whole','short', 'warp', 'rand', 'zero']:
                        for ans in ['','T1','T2','T3']:
                            make_eval(qer,hmm_c, m, n, method, ans)
    for qer in ['dev', 'eva']:
        for hmm_c in [qer, 'doc']:
            for m in [7]:
                for n in [100]:
                    for method in ['hmmp','short', 'norm']:
                        for ans in ['']:
                            make_eval(qer,hmm_c, m, n, method, ans)
    for qer in ['dev']:
        for hmm_c in ['doc', 'dev']:
            for m in [3,5,7]:
                for n in [100,200]:
                    for method in ['hmmp','short', 'norm']:
                        for ans in ['']:
                            make_eval(qer,hmm_c, m, n, method, ans, hier)
    
    for m in [3,5,7]:
        for n in [100,200]:
            for qer in ['dev']:
                for hmm_c in ['doc', 'dev']:
                    for method in ['hmmp', 'norm']:
                        for ans in ['']:
                            make_eval(qer,hmm_c, m, n, method, ans, hier)
    '''
    for m in [3,5,7]:
        for n in [100,200]:
            for qer in ['dev']:
                for hmm_c in ['doc']:
                    for method in ['hmmp', 'norm']:
                        for ans in ['']:
                            make_eval(qer,hmm_c, m, n, method, ans, hier)

def run_make_table(ans=''):
    for m in [3,5,7]:
        for n in [100,200]:
            hier = 'mult'
            print hier,'unsup',(m,n)
            for method in ['norm','hmmp']:
                make_eval('dev','doc', m, n, method, ans, hier)
            print hier,'rev',(m,n)
            for method in ['norm','hmmp']:
                make_eval('dev','dev', m, n, method, ans, hier)
            hier = 'hier'
            print hier,'unsup',parse_lex('doc',m,n)
            for method in ['norm','hmmp']:
                make_eval('dev','doc', m, n, method, ans, hier)
            print hier,'rev',parse_lex('dev',m,n)
            for method in ['norm','hmmp']:
                make_eval('dev','dev', m, n, method, ans, hier)

def run_make_table_new(ans=''):
    choice_list= {('mult','doc'):'a',('mult','dev'):'b',('hier','doc'):'c',('hier','dev'):'d'}
    for cor in ['dev','doc']:
        print ans, cor, ': mn, uv, norm, hmmp'
        for hier in ['mult','hier']:
            if hier == 'hier':
                print '\\multirow{7}{*}{Hier.}'
            else:   
                print '\\multirow{7}{*}{Multi.}'
            for m in [3,5,7]:
                for n in [100,200]:
                    _, _, mn, uv, ev_norm = make_eval('dev', cor, m, n, 'norm', ans, hier)
                    _, _, mn, uv, ev_hmmp = make_eval('dev', cor, m, n, 'hmmp', ans, hier)
                    if hier=='mult':
                        uv = '-'   
                    print '&{:<12}&{:<12}&{:.4f}  &{:.4f}\\\\'.format(mn, uv, ev_norm, ev_hmmp)
            if ans =='':
                avg_method = ','.join(['avg',choice_list[hier,cor], 'norm'])
                _, _, _, _, ev_norm = make_eval('dev', '', '', '', avg_method, ans, '')
                avg_method = ','.join(['avg',choice_list[hier,cor], 'hmmp'])
                _, _, _, _, ev_hmmp = make_eval('dev', '', '', '', avg_method, ans, '')
                print '&{:<12}&{:<12}&{:.4f}  &{:.4f}\\\\ \\hline'.format('average', '-', ev_norm, ev_hmmp)
            
def run_make_table_T():
    choice_list= {('mult','doc'):'a',('mult','dev'):'b',('hier','doc'):'c',('hier','dev'):'d'}
    for cor in ['dev','doc']:
        print cor, ': mn, uv, T1, T2, T3'
        for hier in ['mult','hier']:
            if hier == 'hier':
                print '\\multirow{7}{*}{Hier.}'
            else:   
                print '\\multirow{7}{*}{Mult.}'
            for m in [3,5,7]:
                for n in [100,200]:
                    ev = [0,0,0]
                    for i,ans in enumerate(['T1','T2','T3']):
                        _, _, mn, uv, ev[i] = make_eval('dev', cor, m, n, 'hmmp', ans, hier)
                    if hier=='mult':
                        uv = '-'   
                    print '&{:<12}&{:<12}&{:.4f}  &{:.4f}  &{:.4f}\\\\'.format(mn, uv, ev[0],ev[1],ev[2])
            ev = [0,0,0]
            for i,ans in enumerate(['T1','T2','T3']):
                avg_method = ','.join(['avg',choice_list[hier,cor], 'hmmp'])
                _, _, _, _, ev[i] = make_eval('dev', '', '', '', avg_method, ans, '')
            print '&{:<12}&{:<12}&{:.4f}  &{:.4f}  &{:.4f}\\\\ \\hline'.format('average', '-',ev[0],ev[1],ev[2])

    
def make_score_avg(qer_list, cor_list, m_list, n_list, method_list, hier_list, avg_method_name):
    tok_name = token_name('', '', '', '')
    scr_name = score_name(qer_list[0], tok_name, avg_method_name)
    try:
        sc = pickle.load(open(score(scr_name),'r'))
        print 'loaded precomputed score for ' + scr_name
        return sc
    except:
        print 'computing score for ' + scr_name

    sc = np.zeros(make_score(qer_list[0], cor_list[0], m_list[0], n_list[0], method_list[0], hier_list[0]).shape)
    for qer, cor, m,  n, method, hier in zip(qer_list, cor_list, m_list, n_list, method_list, hier_list):
        sc+= make_score(qer, cor, m, n, method, hier)
    print sc/len(qer_list)
    with open(score(scr_name),'w') as f:
        pickle.dump(sc, f, protocol=pickle.HIGHEST_PROTOCOL)
    return sc

def run_average(ans):
    qer = 'dev'
    #choice_list= [('A','mult','doc'),('B','mult','dev'),('C','hier','dev')]
    #abc_list = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]
    choice_list= [('a','mult','doc'),('b','mult','dev'),('c','hier','doc'),('d','hier','dev')]
    abc_list = [[0],[1],[2],[3]]
    for method in ['norm','hmmp']:
        for abc in abc_list:
            avg_list = []
            alpha_list = ''
            for i in abc:
                alpha, hier,hmm_c = choice_list[i]
                for m in [3,5,7]:
                    for n in [100,200]:
                        avg_list.append((qer,hmm_c,m,n,method,hier))
                alpha_list += alpha
            avg_list = zip(*avg_list)
            avg_method = ','.join(['avg',alpha_list, method])
            avg_list.append(avg_method)
            print avg_method
            make_score_avg(*avg_list)
            make_eval(qer,'', '', '', avg_method, ans, '')


if __name__=='__main__':
    #make_init()
    #ran_qer_token()
    #ran_doc_token()
    #run_doc_token()
    #ran_make_pdist() 
    #ran_make_decode()
    
    #run_make_score()
    #run_make_eval()

    #make_hier('dev',3,10) 
    #run_qer_token()
    #ran_make_pdist('hier') 
    #ran_make_decode('hier')
    
    #run_make_score('hier')
    #run_make_eval('hier')
    #run_make_table('')
    #run_make_table('T1')
    #run_make_table('T2')
    #run_make_table('T3')
    #run_average('') 
    #run_average('T1') 
    #run_average('T2') 
    #run_average('T3') 

    run_make_table_new('')
    run_make_table_T()
