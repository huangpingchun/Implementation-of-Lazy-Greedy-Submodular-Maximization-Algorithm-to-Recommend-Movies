import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def read_Data():
    start = time.time()
    DataFileIn = open("ratings.dat", "r")
    DataList = DataFileIn.readlines()
    A = []
    
    for i in range(len(DataList)):
        a = DataList[i].split("::")
        a = a[0:3]
        a = map(float, a)
        A.append(a)
    
    df = pd.DataFrame(A,columns=['User','Movie','Ratings']) 
    
    
    y = df.pivot(index ='User', columns ='Movie', values ='Ratings')
    y = y.fillna(0)
    
    Final = y.as_matrix(columns=None)
    
    end = time.time() - start
    print "Time taken to load the data is", end, "seconds \n"
    return Final

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros



def plotter(Y,K):
    n_users = Y.shape[0]
    
    delta = []
    
    for i in range(K): 
        Y_dash = Y[:,0:i+1].max(axis = 1)
        Sum_Y = Y_dash.sum()/n_users
        
        delta.append(Sum_Y)
    
    return delta


def delta_function(X,A):
    X_dash = X.max(axis = 0)
    Sum_X_dash = X_dash.sum()/len(X)
    
    a = np.array(A).max(axis = 0)
    Sum_a = a.sum()/len(X)
    
    v = Sum_X_dash - Sum_a
                
    return v


def greedy(X,K):
    n_users = X.shape[0]
    
    A = zerolistmaker(n_users)
    b_val = []
    
    TIME = []
    
    for k in range(K):
        
        s_t = time.time()
        
        if (K == 0):
            value = []
            for j in range(X.shape[1]):
                a_dash = X[:,j]
                #a_dash = a_dash.max(axis = 1)
                Sum_a_dash = a_dash.sum()/n_users
            
                a = np.array(A).max(axis = 0)
                Sum_a = a.sum()/n_users
            
                v = Sum_a_dash - Sum_a
            
                value.append(v)    
            
            b = np.argmax(value)
            
            b_val.append(b)
            
            A = (X[:,b])
            
        
            
        else:
            value = []
            for j in range(X.shape[1]):
                a_dash = np.c_[A, X[:,j]]
                a_dash = a_dash.max(axis = 1)
                Sum_a_dash = a_dash.sum()/n_users
                
                a = np.array(A).max(axis = 0)
                Sum_a = a.sum()/n_users
                
                v = Sum_a_dash - Sum_a
                
                value.append(v)
                
            
            b = np.argmax(value)
            
            b_val.append(b)
            
            A = np.c_[A, (X[:,b])]
        
        e_t = time.time()-s_t
        TIME.append(e_t)
    
    A = np.delete(A, [0], axis=1)        
    return A,b_val,TIME



def lazy_greedy(X,K):
    n_users = X.shape[0]

    A = zerolistmaker(n_users)
    b_val = []
    
    TIME = []
    
    value = []
    
    s_t = time.time()
    
    for j in range(X.shape[1]):
        a_dash = X[:,j]
        #a_dash = a_dash.max(axis = 1)
        Sum_a_dash = a_dash.sum()/n_users
    
        a = np.array(A).max(axis = 0)
        Sum_a = a.sum()/n_users
    
        v = Sum_a_dash - Sum_a
    
        value.append(v)    
        
    
    v_arg = np.argsort(value)
    
    v_arg = v_arg[::-1] 
    
    b = v_arg[0]
    b_val.append(b)
    A = (X[:,b])

    e_t = time.time()-s_t
    TIME.append(e_t)
    
    for k in range(K-1):
        
        s_t = time.time()
        
        d = delta_function((X[:,v_arg[k+1]]),(A))
        if (value[k+2] < d):
            b = v_arg[k+1]
            b_val.append(b)
            
            A = np.c_[A, (X[:,b])]
            
        elif (value[k+2] > d):
            value[k+1] = d
            v_arg = np.argsort(value)
        
            v_arg = v_arg[::-1]
            b = v_arg[k+1]
            b_val.append(b)
        
            A = np.c_[A, (X[:,b])]
            
        e_t = time.time()-s_t
        TIME.append(e_t)
        
    return A,v_arg,TIME



X = read_Data()
print "Data is loaded!! \n"


K = 50

Reco_movies, Rank, T = greedy(X,K)

Reco_movies1, Rank1, T1 = lazy_greedy(X,K)

delta = plotter(Reco_movies,K)

delta1 = plotter(Reco_movies1,K)

T = list(np.cumsum(T))
T1 = list(np.cumsum(T1))


l = []

for i in range(K):
    l.append(i)

plt.plot(l,delta,'r',l,delta1,'b')
plt.grid()
plt.xlabel('Cardinality K')
plt.ylabel('Objective Function')
plt.title('Objective Function v/s Cardinality')
red_patch = mpatches.Patch(color='red', label='The Greedy Algorithm')
blue_patch = mpatches.Patch(color='blue', label='The Lazy Greedy Algorithm')
plt.legend(handles=[red_patch,blue_patch])


plt.show()

plt.plot(l,T,'r',l,T1,'b')
plt.grid()
plt.xlabel('Cardinality K')
plt.ylabel('Time(s)')
plt.title('Time v/s Cardinality')
red_patch = mpatches.Patch(color='red', label='The Greedy Algorithm')
blue_patch = mpatches.Patch(color='blue', label='The Lazy Greedy Algorithm')
plt.legend(handles=[red_patch,blue_patch])


plt.show()



