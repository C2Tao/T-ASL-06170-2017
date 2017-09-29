import zrst
import zrst.asr
import zrst.util
import os
#def wav(cor):
#    if cor =='eng': return '/home/c2tao/ZRC_revision/eng/preprocess/short_wavs'
#    elif cor =='xit': return '/home/c2tao/ZRC_revision/surprise/preprocess/cut_wavs'
#def mfc_name(cor):
#    if cor =='eng': return '/home/c2tao/ZRC_revision/eng/preprocess/mfc'
#    elif cor=='xit': return '/home/c2tao/ZRC_revision/surprise/preprocess/cut_mfc'


def cor_name(cor):
    return 'eng' if cor=='eng' else 'surprise'
def wav_path(cor):
    return '/home/c2tao/mediaeval_token/abx_corpus/{}'.format(cor)

def token_name(cor, m, n, hier):
    return "_".join([cor, str(m), str(n), hier])

def token_path(token_name):
    #'~/ZRC_revision/eng/zrstExps/mfc_50_3'
    #50,100,300,500
    #3,5,7

    #'~/ZRC_revision/surprise/zrstExps/mfc_50_3'
    #50,100,300,500
    #3,5,7,9
    cor, m, n, hier = token_name.split("_")
    if hier == 'mult':
        if cor =='eng':
            return '/home/c2tao/ZRC_revision/eng/zrstExps/mfc_{}_{}/'.format(str(n), str(m))
        elif cor=='xit':
            return '/home/c2tao/ZRC_revision/surprise/zrstExps/mfc_{}_{}/'.format(str(n), str(m))
    if hier == 'hier':
        return '/home/c2tao/mediaeval_token/abx_token/{}/'.format(token_name)
        
def make_hier(cor, arg_m, arg_n):
    tar_path = target=token_path(token_name(cor, arg_m, arg_n, 'hier'))
    if os.path.isdir(tar_path):
        print 'training finished:', tar_path
        return
    #A = zrst.asr.ASR(corpus=wav_path(cor), target=token_path(token_name(cor, arg_m, arg_n, 'hier')), nState=arg_m)
    A = zrst.asr.ASR(target=tar_path, nState=arg_m)
    A.readASR(token_path(token_name(cor, arg_m, arg_n,'mult')))

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
    
    A.x(3)
    A.a_keep()
    A.a_keep()


def run_make_hier():
    token_args = []
    for c in ['eng']:
    #for c in ['xit']:
        for m in [3, 5, 7]:
            for n in [50, 100, 300, 500]:
                token_args.append((c, m, n))
    print token_args
    zrst.util.run_parallel(make_hier, token_args)

if __name__=='__main__':
    #make_hier('eng',7, 50)
    run_make_hier()

