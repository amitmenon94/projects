import numpy as np
import pandas as pd
from sklearn.model_selection._split import train_test_split
from Extract_GSC_data import *
import math
from sklearn.cluster.k_means_ import KMeans
from sklearn.metrics.regression import mean_squared_error

target = output[:,2]
# print(target.shape)
centers =[]
lamda = 2
alpha = 0.2
BATCH = 10000

def getGX(hx):
    Gx = np.array(hx)
    for i in range(hx.shape[0]):
            Gx[i] = sigmoid(hx[i])
    return Gx

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def getClusters(input_data):
    km = KMeans(n_clusters=30, random_state=0).fit(input_data)
    centers = km.cluster_centers_
#     centers = np.insert(centers, 0, 1, 1) ####=========================== ADD BIAS =====================================##
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

def gradient_desc(train_inp,hx,Weight_final,out):
    Gx = getGX(hx)
    for i in range(200):
        reg = lamda*Weight_final
        sub = np.subtract(Gx,out)
        prod = train_inp.T.dot(sub)
        delta = np.add(prod,reg)
        delta[:] = [x/train_inp.shape[0] for x in delta]
        Weight_final = Weight_final - alpha*delta
    return Weight_final
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


def acc_manual(y_act, y_pred):
    print(y_act.shape,y_pred.shape)
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
    Weight_final = np.ones((centers.shape[0],),dtype=float64)

    erms = []
    for start in range(0,len(train),BATCH):
        end = start+BATCH
        phi_train = getphi(train[start:end],BigSigma_inv)
        phi_train = np.insert(phi_train, 0, 1, 1)
        hx = phi_train.dot(Weight_final)
        hx = hx.astype(np.int64)
        Weight_final = gradient_desc(phi_train,hx,Weight_final,train_out[start:end])
        
    hx = phi_train.dot(Weight_final)
    hx = hx.astype(np.int64) 
    Gx = getGX(hx)
    phi_test = getphi(test, BigSigma_inv)
    phi_test = np.insert(phi_test, 0, 1, 1)
    hx_test = phi_test.dot(Weight_final)
    hx_test = hx_test.astype(np.int64) 
    Gx_test = getGX(hx_test)
    phi_val = getphi(val, BigSigma_inv)
    phi_val = np.insert(phi_val, 0, 1, 1)
    hx_val = phi_val.dot(Weight_final)
    hx_val = hx_val.astype(np.int64) 
    Gx_val = getGX(hx_val)

    print("=========================Logistic Regression using Gradient Descent for Subtracted GSC Data===========================")
    print("ERMS Train",mean_squared_error(train_out[start:end], Gx),"Accuracy",acc_manual(train_out[start:end], Gx))
    print("ERMS Test",mean_squared_error(test_out, Gx_test),"Accuracy",acc_manual(test_out, Gx))
    print("ERMS VAL",mean_squared_error(val_out, Gx_val),"Accuracy",acc_manual(val_out, Gx_val))
#         erms.append(mean_squared_error(train_out[start:end], predict_train))
#         training_acc.append(acc_manual(train_out[start:end], predict_train))
 
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
    Weight_final = np.ones((31,),dtype=float64)

    erms = []
    for start in range(0,len(train),BATCH):
        end = start+BATCH
        phi_train = getphi(train[start:end],BigSigma_inv)
        phi_train = np.insert(phi_train, 0, 1, 1)
        hx = phi_train.dot(Weight_final)
        hx = hx.astype(np.int64)
        Weight_final = gradient_desc(phi_train,hx,Weight_final,train_out[start:end])
        
    hx = phi_train.dot(Weight_final)
    hx = hx.astype(np.int64) 
    Gx = getGX(hx)
    phi_test = getphi(test, BigSigma_inv)
    phi_test = np.insert(phi_test, 0, 1, 1)
    hx_test = phi_test.dot(Weight_final)
    hx_test = hx_test.astype(np.int64) 
    Gx_test = getGX(hx_test)
    phi_val = getphi(val, BigSigma_inv)
    phi_val = np.insert(phi_val, 0, 1, 1)
    hx_val = phi_val.dot(Weight_final)
    hx_val = hx_val.astype(np.int64) 
    Gx_val = getGX(hx_val)

    print("=========================Logistic Regression using Gradient Descent for Concatenated GSC Data===========================")
    print("ERMS Train",mean_squared_error(train_out[start:end], Gx),"Accuracy",acc_manual(train_out[start:end], Gx))
    print("ERMS Test",mean_squared_error(test_out, Gx_test),"Accuracy",acc_manual(test_out, Gx))
    print("ERMS VAL",mean_squared_error(val_out, Gx_val),"Accuracy",acc_manual(val_out, Gx_val))

# run_logisitic_human_sub()
run_gradient_desc_gsc_con()

