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

A = read_Data()
print "Data is loaded!! \n"
