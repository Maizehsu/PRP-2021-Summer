import pandas as pd
import numpy as np
from scipy.interpolate import lagrange
def ploy(s,n,k=5)
y = s[list(range(n-k,n))+list(range(n+1,n+1+k))]
y = y[y.notnull()]
return larange(y.index,list(y))(n)
traj = pd.read_csv('DATASET-A.csv', header=None, usecols=[2,3,4]).iloc[:15]
traj.columns = ['timestamp','lon','lat']

