from __future__ import print_function  # Only needed for Python 2
import numpy as np
import os
#example1
'''
MySystem.stdlist.xml 

<?xml version="1.0" encoding="UTF-8"?>
<stdlist termlist_filename="Query.tlist.xml" indexing_time="1.00" language="mul" index_size="1" system_id="Example STD System">
<detected_termlist termid="sws2013_dev_001" term_search_time="2.00" oov_term_count="1">
<term file="sws2013_00001" channel="1" tbeg="0" dur="10" score="-3.0" decision="NO"/>
<term file="sws2013_00002" channel="1" tbeg="0" dur="10" score="-3.0" decision="NO"/>
...
<term file="sws2013_10762" channel="1" tbeg="0" dur="10" score="-3.0" decision="NO"/>
</detected_termlist>
<detected_termlist termid="sws2013_dev_002" term_search_time="2.00" oov_term_count="1">
<term file="sws2013_00001" channel="1" tbeg="0" dur="10" score="-3.0" decision="NO"/>
...
<term file="sws2013_10761" channel="1" tbeg="0" dur="10" score="-3.0" decision="NO"/>
<term file="sws2013_10762" channel="1" tbeg="0" dur="10" score="-3.0" decision="NO"/>
</detected_termlist>
</stdlist>
'''

#example2
'''
<?xml version='1.0' encoding='UTF-8'?>
<stdlist index_size="1" indexing_time="1.00" language="mul" system_id="Vagrant STD System" termlist_filename="quesst2015_dev.tlist.xml">
<detected_termlist oov_term_count="1" term_search_time="1.00" termid="quesst2015_dev_0320">
<term channel="1" decision="NO" dur="10" file="quesst2015_00026" score="-0.435988" tbeg="0" />
<term channel="1" decision="YES" dur="10" file="quesst2015_00029" score="-0.361111" tbeg="0" />
<term channel="1" decision="NO" dur="10" file="quesst2015_00037" score="-0.424553" tbeg="0" />
...
<term channel="1" decision="YES" dur="10" file="quesst2015_01695" score="-0.388458" tbeg="0" />
<term channel="1" decision="YES" dur="10" file="quesst2015_01697" score="-0.291917" tbeg="0" />
</detected_termlist>
<detected_termlist oov_term_count="1" term_search_time="1.00" termid="quesst2015_dev_0321">
<term channel="1" decision="NO" dur="10" file="quesst2015_00026" score="-0.593922" tbeg="0" />
<term channel="1" decision="NO" dur="10" file="quesst2015_00029" score="-0.540931" tbeg="0" />
<term channel="1" decision="NO" dur="10" file="quesst2015_00037" score="-0.515806" tbeg="0" />
<term channel="1" decision="NO" dur="10" file="quesst2015_00051" score="-0.554644" tbeg="0" />
...
</detected_termlist>
</stdlist>
'''

answ_root = '/home/c2tao/mediaeval_2015/data/scoring_quesst2015_lang_noise/'
dev_list = ['quesst2015_dev_{0:04}'.format(q+1) for q in range(445)]
eva_list = ['quesst2015_eval_{0:04}'.format(q+1) for q in range(447)]
doc_list = ['quesst2015_{0:05}'.format(q+1) for q in range(11662)]

            
def write_mediaeval_score(sys_name, term_file, yesno,  score, qer_list, doc_list, score_file):
    xml_beg = '''<?xml version='1.0' encoding='UTF-8'?>\n<stdlist index_size="1" indexing_time="1.00" language="mul" system_id="{0}" termlist_filename="{1}">\n'''
    qer_beg = '''<detected_termlist oov_term_count="1" term_search_time="1.00" termid="{0}">\n'''
    scoring = '''<term channel="1" decision="{0}" dur="10" file="{1}" score="{2}" tbeg="0" />\n'''
    qer_end = '''</detected_termlist>\n'''
    xml_end = '''</stdlist>\n'''

    with open(score_file,'w') as f:
        f.write(xml_beg.format(sys_name, term_file))
        for q, qer in enumerate(qer_list):
            f.write(qer_beg.format(qer))
            for d, doc in enumerate(doc_list):
                dec = 'YES' if yesno[q, d] else 'NO'
                f.write(scoring.format(dec, doc, score[q, d]))
            f.write(qer_end)
        f.write(xml_end)

def write_mediaeval_scores(sys_name, yesno, score_list, answ_path):

    tlist = ''

    stdlist = sys_name + '.stdlist.xml'
    stdpath = answ_root+sys_name
    txtpath = answ_root+'temp.txt'
    for f in os.listdir(answ_root + answ_path):
        if 'tlist.xml' in f:
            tlist = f
    if 'dev' in tlist:
        #print 'using dev_query'
        qer_list = dev_list
        qer = 'dev'
    elif 'eval' in tlist:
        #print 'using eval_query'
        qer_list = eva_list
        qer = 'eva'
    try:
        os.mkdir(stdpath)
    except:
        print('warning', stdpath,'exists')

    #write_mediaeval_score(sys_name, '', yesno, score_list, qer_list, doc_list, stdpath +'/'+stdlist)
    write_mediaeval_score(sys_name, tlist, yesno, score_list, qer_list, doc_list, stdpath +'/'+stdlist)
    
    import subprocess 
    def single():
        subprocess.Popen(['./score-TWV-Cnxe.sh', sys_name, answ_path], cwd = answ_root)
    def multi():

        print(stdpath, file=open(txtpath,'w'))
        subprocess.Popen(['gzip', stdlist],cwd=stdpath)
        subprocess.Popen(['./score-systems_T.sh', txtpath],cwd=answ_root)
    single() 

if __name__=='__main__':

    qer_list = dev_list
    qer_list = eva_list
    
    score = np.random.rand (len(qer_list), len(doc_list)) 
    #print score.shape
    #score = np.zeros([len(qer_list), len(doc_list)]) 
    yesno = np.zeros([len(qer_list), len(doc_list)]) 
    #write_mediaeval_score('sys', 'termfile', yesno, score, qer_list, doc_list, 'temp.stdlist.xml')
    #answ_path = 'groundtruth_quesst2015_dev' 
    answ_path = 'groundtruth_quesst2015_eval' 
    write_mediaeval_scores('kkk', yesno, score, answ_path)





