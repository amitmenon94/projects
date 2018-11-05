import numpy as np
import pandas as pd
from Extract_GSC_data import *
from sklearn.model_selection._split import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.cluster.k_means_ import KMeans
import math
from array import array


target = output[:,2]


# print(target.shape)



def process_human_data():
    subtracted_data = extract_gsc_data_sub()
    subtracted_data = pd.DataFrame(subtracted_data)
    BigSigma = subtracted_data.cov()
    subtracted_data = subtracted_data.loc[:,(BigSigma!=0).any(axis=0)]
    BigSigma = pd.DataFrame(subtracted_data).cov()
    BigSigma = np.diag(np.diag(BigSigma))
    BigSigma_inv = np.linalg.inv(BigSigma)
    subtracted_data = subtracted_data.values
    print(subtracted_data.shape, BigSigma_inv.shape)
    train, test_and_val, train_out, test_and_val_out = train_test_split(subtracted_data , target, test_size=0.3, shuffle=True)
    train = np.array(train)
    pivot = int(len(test_and_val)/2)
    test = test_and_val[:pivot]
    val = test_and_val[pivot:]
   
    test_out = test_and_val_out[:pivot]
    val_out = test_and_val_out[pivot:]
# # print(len(subtracted_data))

    return train, test,val,train_out,test_out,val_out,BigSigma_inv

 

  
EPOCH = 400
BATCH = 10000
  
  
 
def gradient_desc(train_tar,phi_temp,W):
    lamda = 2
    alpha = 0.02
    for i in range(100):
        reg_Ew = lamda*W
        Wphi = W.T.dot(phi_temp.T)
        diff = np.subtract(train_tar, Wphi.T)
        deltaEd = diff.T.dot(phi_temp)
        deltaEd = -deltaEd.T
        deltaEd = np.add(deltaEd, reg_Ew)
        big_deltaE = -deltaEd.dot(alpha)
        W = np.add(W,big_deltaE)
#     prediction= W.T.dot(phi_temp.T)
#     prediction= prediction.T
    return W
  
def getClusters(input_data):
    km = KMeans(n_clusters=10, random_state=0).fit(input_data)
    centers = km.cluster_centers_
    np.insert(centers, 0, 1, 1) ####=========================== ADD BIAS =====================================##
    print("Centers : ",centers.shape)
    return centers
  
def GetScalar(DataRow,MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = BigSigInv.dot(R.T)  
    L = R.dot(T)
    return L
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x
  
  
def getphi(Input_train,BigSigma_inv):
    cent = getClusters(Input_train)
    row = len(Input_train)
    column = len(cent);
    print(row, column)
    phi = pd.DataFrame(0,index=range(row),columns=range(column), dtype='float64')
    phi = phi.values
    for i in range(0,column):
        for j in range(0,row):
            phi[j][i]= GetRadialBasisOut(Input_train[j], cent[i], BigSigma_inv)
    print(phi.shape)
    return phi

def acc_manual(y_act, y_pred):
    print(y_act.shape,y_pred.dtype)
    sum = 0.0
    accuracy = 0.0
    count = 0.0
    for i in range(len(y_pred)):
        if((np.around(y_pred[i], 0)) == y_act[i]):
            count+=1
    print(count)
    accuracy = (float((count*100))/float(len(y_pred)))
    return accuracy  


def run_logisitic_human_sub():
    train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data()

    Weight_final = np.ones((10,),dtype=float64)
    erms = []
    training_acc = []
    for start in range(0,len(train),BATCH):
        end = start+BATCH
        phi_train = getphi(train[start:end],BigSigma_inv)
        Weight_final = gradient_desc(train_out[start:end],phi_train,Weight_final)
        predict_train = phi_train.dot(Weight_final)
        erms.append(mean_squared_error(train_out[start:end], predict_train))
        training_acc.append(acc_manual(train_out[start:end], predict_train))
    print(Weight_final.shape)
    phi_test = getphi(test,BigSigma_inv)
    phi_val = getphi(val, BigSigma_inv)
    predict_val = phi_val.dot(Weight_final)
    predict_test = phi_test.dot(Weight_final)
    print("=========================Gradient Descent for Subtracted GSC Data===========================")
    print("ERMS Train",np.mean(erms),"Accuracy",np.mean(training_acc))
    print("ERMS Test",mean_squared_error(test_out, predict_test),"Accuracy",acc_manual(test_out, predict_test))
    print("ERMS VAL",mean_squared_error(val_out, predict_val),"Accuracy",acc_manual(val_out, predict_val))


def process_gsc__data_con():
    concatenated_data = extract_gsc_data_con()
    concatenated_data = pd.DataFrame(concatenated_data)
    BigSigma = concatenated_data.cov()
    concatenated_data = concatenated_data.loc[:,(BigSigma!=0).any(axis=0)]
    BigSigma = pd.DataFrame(concatenated_data).cov()
    BigSigma = np.diag(np.diag(BigSigma))
    BigSigma_inv = np.linalg.inv(BigSigma)
    concatenated_data = concatenated_data.values
    print(concatenated_data.shape, BigSigma_inv.shape)
    train, test_and_val, train_out, test_and_val_out = train_test_split(concatenated_data , target, test_size=0.3, shuffle=True)
    train = np.array(train)
    pivot = int(len(test_and_val)/2)
    test = test_and_val[:pivot]
    val = test_and_val[pivot:]
   
    test_out = test_and_val_out[:pivot]
    val_out = test_and_val_out[pivot:]
# # print(len(concatenated_data))

    return train, test,val,train_out,test_out,val_out,BigSigma_inv

def run_gradient_desc_gsc_con():
    train,test,val,train_out,test_out,val_out,BigSigma_inv = process_gsc__data_con()
    Weight_final = np.ones((11,),dtype=float64)
    erms = []
    training_acc = []
    for start in range(0,len(train),BATCH):
        end = start+BATCH
        phi_train = getphi(train[start:end],BigSigma_inv)
        Weight_final = gradient_desc(train_out[start:end],phi_train,Weight_final)

    predict_train = phi_train.dot(Weight_final)
    erms.append(mean_squared_error(train_out[start:end], predict_train))
    training_acc.append(acc_manual(train_out[start:end], predict_train))  
    print(Weight_final.shape)
    phi_test = getphi(test,BigSigma_inv)
    phi_val = getphi(val, BigSigma_inv)
    predict_val = phi_val.dot(Weight_final)
    predict_test = phi_test.dot(Weight_final)
    print("=========================Linear Regression using Gradient Descent for Concatenated GSC Data===========================")
    print("ERMS Train",np.mean(erms),"Accuracy",np.mean(training_acc))
    print("ERMS Test",mean_squared_error(test_out, predict_test),"Accuracy",acc_manual(test_out, predict_test))
    print("ERMS VAL",mean_squared_error(val_out, predict_val),"Accuracy",acc_manual(val_out, predict_val))

# run_logisitic_human_sub()
#run_gradient_desc_gsc_con()
