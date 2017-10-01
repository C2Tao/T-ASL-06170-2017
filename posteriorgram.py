import numpy as np

class Posterior(object):
    def __init__(self, lat_path=None):
        self.N = None # size of vocab, including <S>, </S>
        self.T = None # number of frames, 0.01 sec per frame
        self.posterior = None #T x N
        self.lat_path = None #lat_path
        self.lex_path = None
        if lat_path:
            self.posterior = self.read_lat(lat_path) # parse posterior
        
    def normalize(self, posterior):
        return posterior / np.sum(posterior, axis=1, keepdims=True)

    def read_lat(self, lat_path):
        #self.node_list = []
        #self.edge_list = []
        t_list = []
        w_list = []
        with open(lat_path,'r') as f:
            for line in (x.strip() for x in f):
                if line[0]=='v': 
                    if not self.lex_path:
                        lex_path = line.split('=')[1]
                        self.read_lex(lex_path)
                elif line[0]=='U':
                    self.feat_path = line.split('=')[1]
                elif line[0]=='N':
                    pass
                    #self.node_num = int(line.split()[0].split('=')[1])
                    #self.edge_num = int(line.split()[1].split('=')[1])
                elif line[0]=='I':
                    #I = int(line.split()[0].split('=')[1])
                    t = int(float(line.split()[1].split('=')[1])*100)
                    W = line.split()[2].split('=')[1]
                    t_list.append(t)
                    w_list.append(self._lex_table[W])
                    self.T = t_list[-1]
                    #print I, t, W
                    #self.node_list.append((t,W))
                elif line[0]=='J':
                    if line[2] =='0': #'J=0'
                        posterior = np.zeros([self.T, self.N],dtype=np.float32)
                    #J = int(line.split()[0].split('=')[1])
                    S = int(line.split()[1].split('=')[1])
                    E = int(line.split()[2].split('=')[1])
                    a = float(line.split()[3].split('=')[1])
                    l = float(line.split()[4].split('=')[1])

                    posterior[t_list[S]:t_list[E], w_list[E]] += 1#/(t_list[E]-t_list[S])
                    #print J,S,E,a,l
                    #self.edge_list.append((S,E,a,l))
                else:
                    pass
            #assert len(self.node_list)==self.node_num
            #assert len(self.edge_list)==self.edge_num
        self.lat_path = lat_path
        return self.normalize(posterior)

    def read_lex(self, lex_path):
        print 'reading lexicon from:',lex_path
        #self.lex_list = []
        self._lex_table = {}
        self._lex_table['!NULL']=0
        with open(lex_path,'r') as f:
            for i, line in enumerate(x.strip() for x in f):
                #self.lex_list.append(line.split()[0])
                self._lex_table[line.split()[0]] = i
        self.N = len(self._lex_table.keys())-1
        self.lex_path = lex_path
if __name__ =='__main__':
    lat_path1 = '/home/c2tao/mediaeval_token/abx_token/eng_3_300_hierpost/result/s0101a_001.lat'
    lat_path2 = '/home/c2tao/mediaeval_token/abx_token/eng_3_300_hierpost/result/s0101a_002.lat'
    A = Posterior()

    A.posterior = A.read_lat(lat_path1)
    A.posterior = A.read_lat(lat_path2)
    
    #print A.N, A.T
    #print A._lex_table
    #print np.sum(A.posterior, axis = 1).shape
    import matplotlib.pyplot as plt
    plt.matshow(A.posterior)
    plt.show()
