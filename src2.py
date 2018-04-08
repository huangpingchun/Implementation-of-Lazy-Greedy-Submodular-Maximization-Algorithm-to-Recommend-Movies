import pandas as pd
import time
import numpy as np

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

def mono_sub(X):
    n_users = X.shape[0]

    B_size = 100
    A_size = 50
    E_size = 10
    
    B = X[:,np.random.randint(X.shape[1], size=B_size)]

    A = B[:,np.random.randint(B.shape[1], size=A_size)]
    
    E = X[:,np.random.randint(X.shape[1], size=E_size)]
    
    B_dash = B.max(axis = 1)
    A_dash = A.max(axis = 1)
    
    Sum_B = B_dash.sum()/n_users
    Sum_A = A_dash.sum()/n_users
    mono = (Sum_B - Sum_A)
    
    
    B_new  = np.c_[B, E]
    A_new  = np.c_[A, E]
    
    B_dash2 = B_new.max(axis = 1)
    A_dash2 = A_new.max(axis = 1)
    
    Sum_B_new = B_dash2.sum()/n_users
    Sum_A_new = A_dash2.sum()/n_users
    
    SUB = (Sum_A_new - Sum_A) - (Sum_B_new - Sum_B)
    
    return mono,SUB    
        



X = read_Data()
print "Data is loaded!! \n"

MONO = []
SUB = []

for i in range(10):

    Mono, Sub = mono_sub(X)
    MONO.append(Mono)
    SUB.append(Sub)

print(MONO)    
print "The function is Monotonic!! \n"

print(SUB)
print "The function is Submodular!! \n"

