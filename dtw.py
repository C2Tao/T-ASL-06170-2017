import numpy as np
from pprint import pprint
#from numba import jit,autojit
#X = np.array([[.1,.2],[.2,.1]])
#Y = np.array([[.5,.5],[.1,.201],[.2,.1],[.2,.2]])


#@jit
def feature_dtw(X, Y):
    return warp(dist(X, Y))

def pattern_dtw(X, Y, S):
    return warp(soft(X, Y, S))

def soft(X, Y, S):
    #S = ph_dist.get_dm(tok)
    D = np.zeros([len(X),len(Y)])
    for i in range(len(X)):
        for j in range(len(Y)):
            D[i,j] = S[X[i],Y[j]]
    return D

#@jit
def dist(X, Y):
    assert(X.shape[1]==Y.shape[1])
    D = np.zeros([X.shape[0],Y.shape[0]])
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            D[i,j] = np.sqrt(np.sum((X[i]-Y[j])**2))
    return D

#print dist(X,Y)
#@jit
def warp(D,no_path=True):
    A = np.zeros([D.shape[0]+1,D.shape[1]+1])
    L = np.zeros([D.shape[0]+1,D.shape[1]+1],dtype='i4')
    C = np.zeros([D.shape[0]+1,D.shape[1]+1],dtype=('i4,i4'))
    A[:,0] = np.inf
    #print D.shape
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            pre = [(i,j),(i,j+1),(i+1,j)]
            temp = []
            for p in pre:
                temp.append((A[p]+D[i,j])/(L[p]+1))
            idx = np.argmin(temp)
            #idx = np.argmin(map(lambda x: (A[x]+D[i,j])/(L[x]+1),pre))
            best = pre[idx]
            #print i,j,pre,best,idx
            A[i+1,j+1] = A[best]+D[i,j]
            L[i+1,j+1] = L[best]+1
            C[i+1,j+1] = best
    A[1:,1:] /= L[1:,1:]
    best_end = (D.shape[0], np.argmin(A[-1,:]))
    #print best_end
    amin_dist = A[best_end]
    if no_path: return amin_dist
    ix = best_end
    best_path = []
    #pprint(C)
    for p in range(L[best_end]):
        #print ix
        best_path.append((ix[0]-1,ix[1]-1))
        ix = C[tuple(ix)]
    return amin_dist, best_path[::-1]
#D= dist(X,Y)
#print warp(D)

#D = np.random.rand(100,1000)
#for i in range(10):
#    print i
#    warp(D)
