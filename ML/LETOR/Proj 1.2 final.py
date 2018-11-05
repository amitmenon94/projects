import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math



data_input = pd.read_csv("Querylevelnorm_x.csv", header=None)
df2 = pd.read_csv("Querylevelnorm_t.csv", header=None)
data_out = df2.values

####print("Data output", data_out[:10],"\n", data_out.shape)
####print("Data Input", data_input[:10],"\n", data_input.shape, data_input.info())
####
####BigSigma = data_input.cov() #showed that variance cannot be calculated for some_columns
####BigSigma.to_csv('BigSigmaBeforeDeletion.csv')

data_inp = data_input.drop([5,6,7,8,9], axis=1) # delete those columns




##-------------------------------------------------------------------------------------------------------

####split the data here 

train_len = int(math.ceil(len(data_inp)*(80*0.01)))
X_train = data_inp[:train_len]
X_test_and_val = data_inp[train_len:]
pivot = int(len(X_test_and_val)/2)
X_Val = X_test_and_val[:pivot]
X_test = X_test_and_val[pivot:]
y_train = data_out[:train_len]
y_test_and_val = data_out[train_len:]
y_Val = y_test_and_val[:pivot]
y_test = y_test_and_val[pivot:]

##print("X Train: \n",X_train, "\n Y Train: \n",y_train, "\n X VAL: \n", X_Val, "\n Y VAL: \n",
##      y_Val,"\n X test: \n",X_test,"\n Y test: \n", y_test)
##
##print(X_train.shape, X_test_and_val.shape, y_train.shape, y_test_and_val.shape, X_Val.shape,
##      X_test.shape,y_test_and_val.shape)


####------------------------------#Covariance#-------------------------------------------------------------------
BigSigma = data_inp.cov()
BigSigma = np.diag(np.diag(BigSigma))
BigSigma_inv = np.linalg.inv(BigSigma)
#------------------------------------------- Find Clusters ------------------------------------------------
def getClusters(input_data):
    km = KMeans(n_clusters=10, random_state=0).fit(input_data)
    centers = km.cluster_centers_
    print("Centers : ",centers.shape)
    return centers

reg_lamda = 15
reg_lamda = pd.to_numeric(reg_lamda)
#-----------------------------------------------------------------------------------------------------------
def GetScalar(DataRow,MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = BigSigInv.dot(R.T)  
    L = R.dot(T)
    return L
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x


def getphi(Input_train):
    Input_train = Input_train.values
    cent = getClusters(Input_train)
    row = len(Input_train)
    column = len(cent);
    print(row, column)
    phi = pd.DataFrame(0,index=range(row),columns=range(column), dtype='float64')
    phi = phi.values
    for i in range(0,column):
        for j in range(0,row):
            phi[j][i]= GetRadialBasisOut(Input_train[j], cent[i], BigSigma_inv)
    return phi


phi_train = getphi(X_train)
phi_val = getphi(X_Val)
phi_test = getphi(X_test)
##phi_test = getphi(X_test)
print("\nTraining PHI : \n",phi_train.shape,"\n")
print("Training PHI : ",phi_train)
print("\Validation PHI : \n",phi_val.shape,"\n")
print("Validation PHI : ",phi_val)
print("\Testing PHI : \n",phi_test.shape,"\n")
print("Testing PHI : ",phi_test)


#------------------------------------------CalCulate Hypothesis-------------------------------------
def getW(phi,train):
    phiTphi = phi.T.dot(phi)
    ident = np.identity(phiTphi.shape[0])
    regularize = np.dot(reg_lamda,ident)
    phiTphi_plusreg = np.add(regularize,phiTphi)
    phiTphi_reg_inv = np.linalg.inv(phiTphi_plusreg)
    fac= pd.DataFrame(phiTphi_reg_inv.dot(phi.T))
    W= np.dot(fac,train)
    return W
##----------------------------------Accuracy------------------------------------------------------------

def acc_manual(y_act, y_pred):
    print(y_act.shape,y_pred.shape)
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
print("\n W : ",hypothesis)
##-------------------------------------------------------------------------------Estimations---------------------------------
# 
# prediction= hypothesis.T.dot(phi_train.T)
# prediction= prediction.T
# prediction_val= hypothesis.T.dot(phi_val.T)
# prediction_val= prediction_val.T
# prediction_test= hypothesis.T.dot(phi_test.T)
# prediction_test= prediction_test.T
# 
# print("Mean square error for Training: ",mean_squared_error(y_train, prediction))
# print ("Accuracy Manual for Training:",acc_manual(y_train, prediction))
# print("Mean square error for Validation: ",mean_squared_error(y_Val, prediction_val))
# print ("Accuracy Manual for Validation:",acc_manual(y_Val, prediction_val))
# print("Mean square error for Testing: ",mean_squared_error(y_test, prediction_test))
# print ("Accuracy Manual for Testing:",acc_manual(y_test, prediction_test))



#################################################################Gradient Descent########################################################
 

def gradient_desc(train_tar,phi_temp):
    hyp_cuurent = np.dot(220, hypothesis)
    lamda = 2
    alpha = 0.02
    for i in range(600):
        reg_Ew = lamda*hyp_cuurent
        Wphi = hyp_cuurent.T.dot(phi_temp.T)
        diff = np.subtract(train_tar, Wphi.T)
        deltaEd = diff.T.dot(phi_temp)
        deltaEd = -deltaEd.T
        deltaEd = np.add(deltaEd, reg_Ew)
        big_deltaE = -deltaEd.dot(alpha)
        hyp_cuurent = np.add(hyp_cuurent,big_deltaE)
        prediction= hyp_cuurent.T.dot(phi_temp.T)
        prediction= prediction.T
        if(i%10==0):
            print(" W in ", i, "iteration",hyp_cuurent[i])
            
        return prediction

predict_train = gradient_desc(y_train,phi_train)
predict_val = gradient_desc(y_Val,phi_val)
predict_test = gradient_desc(y_test,phi_test)
print("Mean square error for Training: ",mean_squared_error(y_train, predict_train))

print ("Accuracy Manual for Training:",acc_manual(y_train, predict_train))
print("Mean square error for Validation: ",mean_squared_error(y_Val, predict_val))

print ("Accuracy Manual for Validation:",acc_manual(y_Val, predict_val))

print("Mean square error for Testing: ",mean_squared_error(y_test, predict_val))

print ("Accuracy Manual for Testing:",acc_manual(y_test, predict_val))


##################################################################################################################################################
####grid search
##
####clf = SGDClassifier(loss="squared_loss", penalty="l2", max_iter=1500)
####
####param_grid = {
####    'loss': ['squared_loss'],
####    'penalty': ['elasticnet'],
####    'alpha': [10 ** x for x in range(-6, 1)],
####    'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
####    
####}
####
####clf_grid = GridSearchCV(estimator=clf, param_grid=param_grid,
####                                    n_jobs=-1, scoring='roc_auc')
####
####clf_grid.fit(X_train, y_train.ravel())
####
####print(clf_grid.score(X_test,y_test))
