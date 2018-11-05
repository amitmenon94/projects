import ExtractHumanData
from ExtractHumanData import output, extract_human_data_sub,\
    extract_human_data_con
import numpy as np
import pandas as pd
from numpy import int64, float64, float32
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import mean_squared_error
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster.k_means_ import KMeans
import math
from Extract_GSC_data import extract_gsc_data_con

target = output[:,2]
BATCH = 100
def process_human_data(value):
    if (value==0):
        fetched_data = extract_human_data_sub()
    else:
        fetched_data = extract_human_data_con()
    fetched_data = pd.DataFrame(fetched_data)
    BigSigma = fetched_data.cov()
    fetched_data = fetched_data.loc[:,(BigSigma!=0).any(axis=0)]
    BigSigma = pd.DataFrame(fetched_data).cov()
    BigSigma = np.diag(np.diag(BigSigma))
    BigSigma_inv = np.linalg.inv(BigSigma)
    fetched_data = fetched_data.values
    print(fetched_data.shape, BigSigma_inv.shape)
    train, test_and_val, train_out, test_and_val_out = train_test_split(fetched_data , target, test_size=0.3, shuffle=True)
    train = np.array(train)
    pivot = int(len(test_and_val)/2)
    test = test_and_val[:pivot]
    val = test_and_val[pivot:]
    
    test_out = test_and_val_out[:pivot]
    val_out = test_and_val_out[pivot:]
 # print(len(fetched_data))
 
    return train, test,val,train_out,test_out,val_out,BigSigma_inv


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


def gradient_desc(train_tar,train_inp,W):
    lamda = 2
    alpha = 0.2
    for i in range(600):
        reg_Ew = lamda*W
        Wphi = W.T.dot(train_inp.T)
        diff = np.subtract( Wphi.T,train_tar)
        deltaEd = diff.T.dot(train_inp)
        deltaEd = deltaEd.T
        deltaEd = np.add(deltaEd, reg_Ew)
        deltaEd[:]= [x/train_inp.shape[0] for x in deltaEd]
        big_deltaE = deltaEd.dot(alpha)
        W = np.subtract(W,big_deltaE)
        return W
 
 
def acc_manual(y_act, y_pred):
    sum = 0.0
    count = 0.0
    accuracy = 0.0
    for i in range(len(y_pred)):
        if((np.around(y_pred[i], 0)) == y_act[i]):
            count+=1
    accuracy = (float((count*100))/float(len(y_pred)))
    return accuracy  

 
def process_gsc__data_con(value):
    if (value==0):
        fetched_data = extract_human_data_sub()
    else:
        fetched_data = extract_human_data_con()
    fetched_data = extract_gsc_data_con()
    fetched_data = pd.DataFrame(fetched_data)
    BigSigma = fetched_data.cov()
    fetched_data = fetched_data.loc[:,(BigSigma!=0).any(axis=0)]
    BigSigma = pd.DataFrame(fetched_data).cov()
    BigSigma = np.diag(np.diag(BigSigma))
    BigSigma_inv = np.linalg.inv(BigSigma)
    fetched_data = fetched_data.values
    print(fetched_data.shape, BigSigma_inv.shape)
    train, test_and_val, train_out, test_and_val_out = train_test_split(fetched_data , target, test_size=0.3, shuffle=True)
    train = np.array(train)
    pivot = int(len(test_and_val)/2)
    test = test_and_val[:pivot]
    val = test_and_val[pivot:]
   
    test_out = test_and_val_out[:pivot]
    val_out = test_and_val_out[pivot:]
# # print(len(fetched_data))

    return train, test,val,train_out,test_out,val_out,BigSigma_inv


#---------------------Linear Regression Using Gradient descent With HOD Data : Subtraction code below------------------------------------------------------------------------------------------------------
# train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data(0)
# train = np.insert(train, 0, 1, 1) # Adding the Bias
# val = np.insert(val, 0, 1, 1)
# test = np.insert(test, 0, 1, 1)
#    
#    
#    
# Weight_final = np.ones((train.shape[1]),dtype=float64)
# erms = []
# training_acc = []
# for start in range(0,len(train),BATCH):
#     end = start+BATCH
#     Weight_final = gradient_desc(train_out[start:end],train[start:end],Weight_final)
#        
#    
# predict_train = train[start:end].dot(Weight_final)
# predict_test = test.dot(Weight_final)
# predict_val = val.dot(Weight_final)
# print("-----------------------------Linear Regression Using Gradient descent With HOD Data : Subtraction--------------------------")
# print("ERMS Training ",mean_squared_error(train_out[start:end], predict_train))
# print("Accuracy Training",acc_manual(train_out[start:end], predict_train)) 
# print("ERMS Test",mean_squared_error(test_out, predict_test))
# print("Accuracy Test",acc_manual(test_out, predict_test))
# print("ERMS Validation",mean_squared_error(val_out, predict_val))
# print("Accuracy Validation",acc_manual(val_out, predict_val))
#     
 
# #----------------------------------Linear Regression Using Gradient descent With HOD Data : Concatenated code below------------------------------------------------------------------------------------------------------
# train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data(0)
# min_max_scaler = preprocessing.MinMaxScaler()
# train = min_max_scaler.fit_transform(train)
# train = np.insert(train, 0, 1, 1) # Adding the Bias
# val = np.insert(val, 0, 1, 1)
# test = np.insert(test, 0, 1, 1)
#     
#     
#     
# Weight_final = np.ones((train.shape[1]),dtype=float64)
# erms = []
# training_acc = []
# for start in range(0,len(train),BATCH):
#     end = start+BATCH
#     Weight_final = gradient_desc(train_out[start:end],train[start:end],Weight_final)
#         
#     
# predict_train = train[start:end].dot(Weight_final)
# predict_test = test.dot(Weight_final)
# predict_val = val.dot(Weight_final)
# print("-----------------------------Linear Regression Using Gradient descent With HOD Data : Concatenation--------------------------")
# print("ERMS Training ",mean_squared_error(train_out[start:end], predict_train))
# print("Accuracy Training",acc_manual(train_out[start:end], predict_train)) 
# print("ERMS Test",mean_squared_error(test_out, predict_test))
# print("Accuracy Test",acc_manual(test_out, predict_test))
# print("ERMS Validation",mean_squared_error(val_out, predict_val))
# print("Accuracy Validation",acc_manual(val_out, predict_val))
# #    
#---------------------------------------------------------------------------------------------------------------------------     
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  
# regr = linear_model.LinearRegression()
#   
# regr.fit(train, train_out)
#  
# predict_train = regr.predict(train)
# predict_test = regr.predict(test)
#   
# print('Coefficients: \n', regr.coef_)
# print("Mean squared error: %.2f"
#       % mean_squared_error(test_out, predict_test))
#  
# print(acc_manual(train_out, predict_train)) 
# print(acc_manual(test_out, predict_test))
# plt.scatter(test[:,0], test[:,1],  color='black')
#  # plt.plot(test[:,1], predict_test, color='blue', linewidth=3)
#  
#  plt.xticks(())
#  plt.yticks(())
#  
# plt.show()