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

def greedy(X,K):
    n_users = X.shape[0]
    
    A = zerolistmaker(n_users)
    b_val = []
    
    
    for k in range(K):
        
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
    
    A = np.delete(A, [0], axis=1)        
    return A,b_val,value


def plotter(Y,K):
    n_users = Y.shape[0]
    
    delta = []
    
    for i in range(K): 
        Y_dash = Y[:,0:i+1].max(axis = 1)
        Sum_Y = Y_dash.sum()/n_users
        
        delta.append(Sum_Y)
    
    return delta


X = read_Data()
print "Data is loaded!! \n"


K = 50
Reco_movies, Rank, val = greedy(X,K)

delta = plotter(Reco_movies,K)

plt.plot(delta,'r')
plt.grid()
plt.xlabel('Cardinality K')
plt.ylabel('Objective Function')
plt.title('Objective Function v/s Cardinality')
red_patch = mpatches.Patch(color='red', label='The Greedy Algorithm')
plt.legend(handles=[red_patch])

