import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.cluster import KMeans
import math


def split_cell(x):
    if x.dtype == object:
        x = x.str.split(':').str[1]
    return(x)    

df = pd.read_csv("Querylevelnorm.csv", header=None)

data_out = df[0].astype(float)


cols = df.columns[2:48]
df[cols] = df[cols].apply(split_cell)
#print(df[cols].dtypes)

data_inp = df[cols].apply(pd.to_numeric, errors='coerce')

data_inp = data_inp.drop([7, 8,9,10,11], axis=1)

BigSigma = data_inp.cov()
BigSigma = np.diag(np.diag(BigSigma))
BigSigma_inv = np.linalg.inv(BigSigma)
print(BigSigma)

x = np.array(data_inp)
y = np.array(data_out).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(
data_inp, data_out, test_size=0.2, random_state=1)

km = KMeans(n_clusters=10, random_state=0).fit(X_train)
centers = km.cluster_centers_
reg_lamda = 0.03
reg_lamda = pd.to_numeric(reg_lamda)
def GetScalar(DataRow,MuRow, BigSigInv):

    R = DataRow.sub(MuRow)
    T = BigSigInv.dot(R.T)  
    L = R.dot(T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x


X_train_T = X_train.T
row = X_train.shape[0]
column = len(centers);
print(row, column)
phi = pd.DataFrame(0,index=range(row),columns=range(column), dtype='float64')
for i in range(0,column):
    for j in range(0,row):
            phi.iloc[j][i]= GetRadialBasisOut(X_train_T.iloc[j], centers[i], BigSigma_inv)

phiTphi= phi.T.dot(phi)
ident = pd.DataFrame(np.identity(phiTphi.shape[0]))
regularize = np.dot(reg_lamda,ident)
phiTphi_plusreg = regularize + phiTphi
phiTphi_reg_inv= pd.DataFrame(np.linalg.inv(phiTphi_plusreg.values))
print(phiTphi_reg_inv.shape)
fac= phiTphi_reg_inv.dot(phi.T)
print(fac.shape , y_train.shape)
W= fac.multiply(y_train)

X_test_T = X_test.T
row = len(X_test)
column = len(centers)
print(row, column)
phi_test = pd.DataFrame(0,index=range(row),columns=range(column), dtype='float64')
for i in range(0,column):
    for j in range(0,row):
            phi_test.iloc[j][i]= GetRadialBasisOut(X_train.iloc[j], centers[i], BigSigma_inv)

##row= X_test.shape[0]
##phi_test= np.empty((row,column), dtype= float)
##for i in range(row):
##	for j in range(column):
##		dist= np.linalg.norm(X_test[i]-centers[j])
##		phi_test[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))
##
prediction= phi.multiply(W)
print(prediction.head(10))
##prediction= 0.5*(np.sign(prediction-0.5)+1)
scores= prediction.sub(y.train)
scores = scores**2
scores = scores/X_test.shape[0]
scores = scores**0.5
print (scores.head(10))

print("Input data shape is", X_train.shape, "\nOutput shape is", y_train.shape)
print("X shape is", x.shape, "\nY shape is", y.shape)
print("centers shape is", centers.shape)
print("Variance shape is", covariance.shape)
print("phi shape is", phi.shape)

