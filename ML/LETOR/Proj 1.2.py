import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.cluster import KMeans
import math


## Format Input
def split_cell(x):
    if x.dtype == object:
        x = x.str.split(':').str[1]
    return(x)

df = pd.read_csv("Querylevelnorm.csv", header=None)
data_out = data_out = df[0].astype(float).values

cols = df.columns[2:48]
df[cols] = df[cols].apply(split_cell)
data_inp = df[cols].apply(pd.to_numeric, errors='coerce')

#BigSigma = data_inp.cov() #showed that variance cannot be calculated for
#some_columns

data_inp = data_inp.drop([7, 8,9,10,11], axis=1) # delete those columns

##------------------------------##-------------------------------------------------------------------
#Find Covariance matrix
BigSigma = data_inp.cov()
BigSigma = np.diag(np.diag(BigSigma))
BigSigma_inv = np.linalg.inv(BigSigma)
print(BigSigma)

#-------------------------------------------------------------------------------------------------------

#split the data here 
X_train, X_test_and_val, y_train, y_test_and_val = train_test_split(
data_inp, data_out, test_size=0.2, random_state=0)
pivot = int(len(X_test_and_val)/2)
X_Val = X_test_and_val[:pivot]
X_test = X_test_and_val[pivot:]
y_Val = y_test_and_val[:pivot]
y_test = y_test_and_val[pivot:]


print(X_train.shape, X_test_and_val.shape, y_train.shape, y_test_and_val.shape, X_Val.shape,
      X_test.shape,y_test_and_val.shape )

#------------------------------------------- Find Clusters ------------------------------------------------

km = KMeans(n_clusters=10, random_state=0).fit(X_train)
centers = km.cluster_centers_
print("Centers : ",centers.shape)
#-----------------------------------------------------------------------------------------------------------
reg_lamda = 0.03
reg_lamda = pd.to_numeric(reg_lamda)
#-----------------------------------------------------------------------------------------------------------
def GetScalar(DataRow,MuRow, BigSigInv):
    R = DataRow.sub(MuRow)
    T = BigSigInv.dot(R.T)  
    L = R.dot(T)
    return L
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x


def getphi(Input_train):
    row = len(Input_train)
    column = len(centers);
    print(row, column)
    phi = pd.DataFrame(0,index=range(row),columns=range(column), dtype='float64')
    for i in range(0,column):
        for j in range(0,row):
            phi.iloc[j][i]= GetRadialBasisOut(Input_train.iloc[j], centers[i], BigSigma_inv)
    return phi


phi_train = getphi(X_train)
##phi_test = getphi(X_test)
print("Training PHI : ",phi_train.shape)

#------------------------------------------CalCulate Hypothesis-------------------------------------
def getW(phi,train):
    train = pd.DataFrame(train)
    print("Train shape", train.shape)
    phiTphi = phi.T.dot(phi)
    print("Shape of PhiTphi={}".format(phiTphi.shape))
    ident = pd.DataFrame(np.identity(phiTphi.shape[0]))
    print("Shape of identity={}".format(ident.shape))
    regularize = np.dot(reg_lamda,ident)
    print("Regularize shape",regularize.shape)
    phiTphi_plusreg = np.add(regularize,phiTphi)
    print("Shape of phiTphi_plusreg={}".format(phiTphi_plusreg.shape))
    phiTphi_reg_inv = pd.DataFrame(np.linalg.inv(phiTphi_plusreg.values))
    print("Shape of phiTphi_reg_inv={}".format(phiTphi_reg_inv.shape))
    print("Shape of phi Transpose{}".format(phi.T.shape))
    fac= pd.DataFrame(phiTphi_reg_inv.dot(phi.T))
    print("FAC shape ",fac.shape)
    W= np.dot(fac,train)
    return W

def acc_manual(y_act, y_pred):
    y_act = pd.DataFrame(y_act)
    print(y_act.shape,y_pred.shape)
    print(y_act.head(20),y_pred.head(20))
    sum = 0.0
    accuracy = 0.0
    count = 0.0
    for i in range(len(y_pred)):
        if(int(np.around(y_pred[i][0], 0)) == y_act[i][0]):
            count+=1
    print(count)
    accuracy = (float((count*100))/float(len(y_pred)))
    return accuracy

hypothesis  = getW(phi_train,y_train)
print("W : ",hypothesis.shape)


prediction= phi_train.dot(hypothesis)
##prediction_test = phi_test.dot(hypothesis)
print(prediction.head(10))


print("Mean square error : ",mean_squared_error(y_train, prediction))
##print("Mean square error for test : ",mean_squared_error(y_test,prediction_test))
##scores= prediction.sub(y_train)
##scores = scores**2
##sum = scores.sum()
##scores = scores/X_train.shape[0]
##scores = scores**0.5
##print ("Accuracy:",accuracy_score(y_train, prediction, normalize=False))

print ("Accuracy Manual:",acc_manual(y_train, prediction))

#--------------------------------------------------------Gradient Descent-----------------------------------------------------------

'''
W_curr = W.multiply(220)



'''
