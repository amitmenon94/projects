import numpy as np
import pandas as pd
from numpy import int64, float64, float32
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import mean_squared_error
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import int64, float64, float32
from cmath import isnan
import math
from sklearn.cluster.k_means_ import KMeans
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.preprocessing.data import StandardScaler

BATCH_HUMAN = 100
BATCH_GSC = 1000
lamda = 0.1
alpha = 0.02
CENTERS = 15
def generate_raw_data(dataset):
    HUMAN_VALUE_RANGE = 791
    GSC_VALUE_RANGE = 30000
    if(dataset==0):
        RANGE = HUMAN_VALUE_RANGE
        data = pd.read_csv("HumanObserved-Features-Data.csv",dtype={
                                                                        'f1':int64,
                                                                        'f2':int64,
                                                                        'f3':int64,
                                                                        'f4':int64,
                                                                        'f5':int64,
                                                                        'f6':int64,
                                                                        'f7':int64,
                                                                        'f8':int64,
                                                                        'f9':int64})

        data.drop("Unnamed: 0", 1, inplace=True)
        same = pd.read_csv("same_pairs.csv") 
        diff = pd.read_csv("diffn_pairs.csv") 
    else:
        RANGE = GSC_VALUE_RANGE
        data = pd.read_csv("GSC-Features.csv")
        same = pd.read_csv("same_pairs_GSC.csv") 
        diff = pd.read_csv("diffn_pairs_GSC.csv")


    p = np.random.permutation(range(RANGE)) 
    data = data.values
    same = same.values
    same = same[p]
    diff = diff.values
    diff = diff[p]
    output=pd.DataFrame(same)
    output = output.append(pd.DataFrame(diff))
    output = output.values
    data_features = data[:,1:10]
    data_features = data_features.astype(float64)
    data_features = np.nan_to_num(data_features)
    return data,same,diff,output,data_features
    
    
def extract_human_data_sub():
    data,same,diff,output,data_features = generate_raw_data(0)
    subtract_features_inp = []
    absolute = []
    target = []
    print("Extracting Subtracted data for HOD")
    for i in range(len(same)):
        temp = data_features[np.where(data[:,0]==same[i][0])]
        temp2 = data_features[np.where(data[:,0]==same[i][1])]
        absolute = abs(temp-temp2)
        subtract_features_inp.append(absolute[0,:])
 
    for i in range(len(diff)):
        temp = data_features[np.where(data[:,0]==diff[i][0])]
        temp2 = data_features[np.where(data[:,0]==diff[i][1])]
        absolute = abs(temp-temp2)
        subtract_features_inp.append(absolute[0,:])
 
    target = output[:,2]  
    print("Done")
    return subtract_features_inp,target

def extract_human_data_con():
    print("Extracting Concatenated data for HOD")
    data,same,diff,output,data_features = generate_raw_data(0)
    absolute = []
    concat_input = []
    target = []
    for i in range(len(same)):
        temp = data_features[np.where(data[:,0]==same[i][0])]
        temp2 = data_features[np.where(data[:,0]==same[i][1])]
        concat = np.concatenate((temp,temp2),axis=1)
        concat_input.append(concat[0,:])
 
    for i in range(len(diff)):
        temp = data_features[np.where(data[:,0]==diff[i][0])]
        temp2 = data_features[np.where(data[:,0]==diff[i][1])]
        concat = np.concatenate((temp,temp2),axis=1)
        concat_input.append(concat[0,:])
    target = output[:,2]  
    print("Done")
    return concat_input,target
def extract_gsc_data_sub():
    data,same,diff,output,data_features = generate_raw_data(1)
    subtract_features_inp = []
    absolute = []
    target = []
    print("Extracting Subtracted data for GSC")
    for i in range(len(same)):
        temp = data_features[np.where(data[:,0]==same[i][0])]
        temp2 = data_features[np.where(data[:,0]==same[i][1])]
        absolute = abs(temp-temp2)
        subtract_features_inp.append(absolute[0,:])
  
    for i in range(len(diff)):
        temp = data_features[np.where(data[:,0]==diff[i][0])]
        temp2 = data_features[np.where(data[:,0]==diff[i][1])]
        absolute = abs(temp-temp2)
        subtract_features_inp.append(absolute[0,:])
         

    target = output[:,2]  
    print("Done")
    return subtract_features_inp,target
 
def extract_gsc_data_con():
    data,same,diff,output,data_features = generate_raw_data(1)
    absolute = []
    concat_input = []
    target = []
    print("Extracting Concatenatedx data for GSC")
    for i in range(len(same)):
        temp = data_features[np.where(data[:,0]==same[i][0])]
        temp2 = data_features[np.where(data[:,0]==same[i][1])]
        concat = np.concatenate((temp,temp2),axis=1)
        concat_input.append(concat[0,:])
 
    for i in range(len(diff)):
        temp = data_features[np.where(data[:,0]==diff[i][0])]
        temp2 = data_features[np.where(data[:,0]==diff[i][1])]
        concat = np.concatenate((temp,temp2),axis=1)
        concat_input.append(concat[0,:])

    target = output[:,2]  
    print("Done")
    return concat_input,target

 

def process_human_data(value):
    if (value==0):
        fetched_data,target = extract_human_data_sub()
    else:
        fetched_data,target = extract_human_data_con()
    fetched_data = pd.DataFrame(fetched_data)
    BigSigma = fetched_data.cov()
    fetched_data = fetched_data.loc[:,(BigSigma!=0).any(axis=0)]
    BigSigma = pd.DataFrame(fetched_data).cov()
    BigSigma = np.diag(np.diag(BigSigma))
    BigSigma_inv = np.linalg.inv(BigSigma)
    fetched_data = fetched_data.values
    print(fetched_data.shape, BigSigma_inv.shape)
    train, test_and_val, train_out, test_and_val_out = train_test_split(fetched_data , target, test_size=0.2, shuffle=True)
    train = np.array(train)
    pivot = int(len(test_and_val)/2)
    test = test_and_val[:pivot]
    val = test_and_val[pivot:]
    
    test_out = test_and_val_out[:pivot]
    val_out = test_and_val_out[pivot:]
 # print(len(fetched_data))
 
    return train, test,val,train_out,test_out,val_out,BigSigma_inv


def process_gsc_data(value):
    if (value==0):
        fetched_data,target = extract_gsc_data_sub()
    else:
        fetched_data,target = extract_gsc_data_con()
    fetched_data = pd.DataFrame(fetched_data)
    BigSigma = fetched_data.cov()
    fetched_data = fetched_data.loc[:,(BigSigma!=0).any(axis=0)]
    BigSigma = pd.DataFrame(fetched_data).cov()
    BigSigma = np.diag(np.diag(BigSigma))
    BigSigma_inv = np.linalg.inv(BigSigma)
    fetched_data = fetched_data.values
    print(fetched_data.shape, BigSigma_inv.shape)
    train, test_and_val, train_out, test_and_val_out = train_test_split(fetched_data , target, test_size=0.2, shuffle=True)
    train = np.array(train)
    pivot = int(len(test_and_val)/2)
    test = test_and_val[:pivot]
    val = test_and_val[pivot:]
    
    test_out = test_and_val_out[:pivot]
    val_out = test_and_val_out[pivot:]
 # print(len(fetched_data))
 
    return train, test,val,train_out,test_out,val_out,BigSigma_inv


def getClusters(input_data):
    km = KMeans(n_clusters=CENTERS, random_state=0).fit(input_data)
    centers = km.cluster_centers_
    return centers
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def getGX(hx):
    Gx = np.array(hx)
    for i in range(hx.shape[0]):
            Gx[i] = sigmoid(hx[i])
    return Gx
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

    phi = pd.DataFrame(0,index=range(row),columns=range(column), dtype='float64')
    phi = phi.values
    for i in range(0,column):
        for j in range(0,row):
            phi[j][i]= GetRadialBasisOut(Input_train[j], cent[i], BigSigma_inv)
    return phi
def gradient_desc(train_inp,hx,Weight_final,out):
    Gx = getGX(hx)
    for i in range(600):
        temp = np.square(Weight_final)
        temp[0] = 0
        reg = lamda*Weight_final
        sub = np.subtract(Gx,out)
        prod = train_inp.T.dot(sub)
        delta = np.add(prod,reg)
        delta[:] = [x/train_inp.shape[0] for x in delta]
        Weight_final = Weight_final - alpha*delta
    return Weight_final
# def gradient_desc(train_tar,train_inp,W):
#     lamda = 2
#     alpha = 0.02
#     for i in range(200):
#         temp = np.square(W)
#         temp[0] = 0
#         reg_Ew = lamda*temp
#         Wphi = W.T.dot(train_inp.T)
#         diff = np.subtract( Wphi.T,train_tar)
#         deltaEd = diff.T.dot(train_inp)
#         deltaEd = deltaEd.T
#         deltaEd = np.add(deltaEd, reg_Ew)
#         deltaEd[:]= [x/train_inp.shape[0] for x in deltaEd]
#         print(deltaEd)
#         big_deltaE = deltaEd.dot(alpha)
#         W = np.subtract(W,big_deltaE)
#         return W
def gradient_desc_with_basis(train_inp,Weight_final,out):
#     lamda = 0.22
#     alpha = 0.02
    for i in range(600):
        temp = np.square(Weight_final)
        temp[0] = 0
        hx = train_inp.dot(Weight_final)
        reg = lamda*temp
        sub = np.subtract(hx,out)
        prod = train_inp.T.dot(sub)
        delta = np.add(prod,reg)
        delta[:] = [x/train_inp.shape[0] for x in delta]
        Weight_final = Weight_final - alpha*delta
    return Weight_final
 
def acc_manual(y_act, y_pred):
    sum = 0.0
    count = 0.0
    accuracy = 0.0
    for i in range(len(y_pred)):
        if((np.around(y_pred[i], 0)) == y_act[i]):
            count+=1
    accuracy = (float((count*100))/float(len(y_pred)))
    return accuracy 


def linear_regress_human_sub(train,test,val,train_out,test_out,val_out,BigSigma_inv):
    #---------------------Linear Regression Using Gradient descent With HOD Data : Subtraction code below------------------------------------------------------------------------------------------------------
    min_max_scaler = preprocessing.MinMaxScaler()
    train = min_max_scaler.fit_transform(train)
    test = min_max_scaler.fit_transform(test)
    val = min_max_scaler.fit_transform(val)
    train = np.insert(train, 0, 1, 1) # Adding the Bias
    val = np.insert(val, 0, 1, 1)
    test = np.insert(test, 0, 1, 1)
        
    Weight_final = np.ones((train.shape[1]),dtype=float64)
    erms = []
    training_acc = []
    for start in range(0,len(train),BATCH_HUMAN):
        end = start+BATCH_HUMAN
        Weight_final = gradient_desc_with_basis(train[start:end],Weight_final,train_out[start:end])
            
        
    predict_train = train[start:end].dot(Weight_final)
    predict_test = test.dot(Weight_final)
    predict_val = val.dot(Weight_final)
    print("-----------------------------Linear Regression Using Gradient descent With HOD Data : Subtraction--------------------------")
    print("ERMS Training ",mean_squared_error(train_out[start:end], predict_train))
    print("Accuracy Training",acc_manual(train_out[start:end], predict_train)) 
    print("ERMS Test",mean_squared_error(test_out, predict_test))
    print("Accuracy Test",acc_manual(test_out, predict_test))
    print("ERMS Validation",mean_squared_error(val_out, predict_val))
    print("Accuracy Validation",acc_manual(val_out, predict_val))       

def linear_regress_human_con(train,test,val,train_out,test_out,val_out,BigSigma_inv):
    min_max_scaler = preprocessing.MinMaxScaler()
    train = min_max_scaler.fit_transform(train)
    test = min_max_scaler.fit_transform(test)
    val = min_max_scaler.fit_transform(val)
    train = np.insert(train, 0, 1, 1) # Adding the Bias
    val = np.insert(val, 0, 1, 1)
    test = np.insert(test, 0, 1, 1)
    Weight_final = np.ones((train.shape[1],),dtype=float64)

    erms = []
    for start in range(0,len(train),BATCH_HUMAN):
        end = start+BATCH_HUMAN

#         hx = phi_train.dot(Weight_final)
#         hx = hx.astype(np.int64)
        Weight_final = gradient_desc_with_basis(train[start:end],Weight_final,train_out[start:end])
        
    predict_train = train[start:end].dot(Weight_final)
    predict_test = test.dot(Weight_final)
    predict_val = val.dot(Weight_final)
    print("-----------------------------Linear Regression Using Gradient descent With HOD Data : Concatenation--------------------------")
    print("ERMS Training ",mean_squared_error(train_out[start:end], predict_train))
    print("Accuracy Training",acc_manual(train_out[start:end], predict_train)) 
    print("ERMS Test",mean_squared_error(test_out, predict_test))
    print("Accuracy Test",acc_manual(test_out, predict_test))
    print("ERMS Validation",mean_squared_error(val_out, predict_val))
    print("Accuracy Validation",acc_manual(val_out, predict_val))  

def run_linear_reg_gsc_sub(train,test,val,train_out,test_out,val_out,BigSigma_inv):
    Weight_final = np.ones((CENTERS+1,),dtype=float64)
    erms = []
    training_acc = []
    for start in range(0,len(train),BATCH_GSC):
        end = start+BATCH_GSC
        phi_train = getphi(train[start:end],BigSigma_inv)
        phi_train = np.insert(phi_train, 0, 1, 1)
        Weight_final = gradient_desc_with_basis(phi_train, Weight_final, train_out[start:end])

    predict_train = phi_train.dot(Weight_final)    
    phi_test = getphi(test,BigSigma_inv)
    phi_test = np.insert(phi_test, 0, 1, 1)
    phi_val = getphi(val, BigSigma_inv)
    phi_val= np.insert(phi_val, 0, 1, 1)
    predict_val = phi_val.dot(Weight_final)
    predict_test = phi_test.dot(Weight_final)
    print("=========================Linear regression Gradient Descent for Subtracted GSC Data===========================")
    print("ERMS Train",mean_squared_error(train_out[start:end], predict_train),"Accuracy",acc_manual(train_out[start:end], predict_train))
    print("ERMS Test",mean_squared_error(test_out, predict_test),"Accuracy",acc_manual(test_out, predict_test))
    print("ERMS VAL",mean_squared_error(val_out, predict_val),"Accuracy",acc_manual(val_out, predict_val))

def run_linear_reg_gsc_con(train,test,val,train_out,test_out,val_out,BigSigma_inv):
    Weight_final = np.ones((CENTERS+1,),dtype=float64)
    erms = []
    training_acc = []
    for start in range(0,len(train),BATCH_GSC):
        end = start+BATCH_GSC
        phi_train = getphi(train[start:end],BigSigma_inv)
        phi_train = np.insert(phi_train, 0, 1, 1)
        Weight_final = gradient_desc_with_basis(phi_train, Weight_final, train_out[start:end])

    predict_train = phi_train.dot(Weight_final)
    phi_test = getphi(test,BigSigma_inv)
    phi_test = np.insert(phi_test, 0, 1, 1)
    phi_val = getphi(val, BigSigma_inv)
    phi_val= np.insert(phi_val, 0, 1, 1)
    predict_val = phi_val.dot(Weight_final)
    predict_test = phi_test.dot(Weight_final)
    print("=========================Linear Regression using Gradient Descent for Concatenated GSC Data===========================")
    print("ERMS Train",mean_squared_error(train_out[start:end], predict_train),"Accuracy",acc_manual(train_out[start:end], predict_train))
    print("ERMS Test",mean_squared_error(test_out, predict_test),"Accuracy",acc_manual(test_out, predict_test))
    print("ERMS VAL",mean_squared_error(val_out, predict_val),"Accuracy",acc_manual(val_out, predict_val))

def run_logistic_human_con(train,test,val,train_out,test_out,val_out,BigSigma_inv):

    Weight_final = np.ones((CENTERS+1,),dtype=float64)

    erms = []
    for start in range(0,len(train),BATCH_HUMAN):
        end = start+BATCH_HUMAN
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

def run_logistic_human_sub(train,test,val,train_out,test_out,val_out,BigSigma_inv):

    Weight_final = np.ones((CENTERS+1,),dtype=float64)

    erms = []
    for start in range(0,len(train),BATCH_HUMAN):
        end = start+BATCH_HUMAN
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

    print("=========================Logistic Regression using Gradient Descent for Subtracted HOD Data===========================")
    print("ERMS Train",mean_squared_error(train_out[start:end], Gx),"Accuracy",acc_manual(train_out[start:end], Gx))
    print("ERMS Test",mean_squared_error(test_out, Gx_test),"Accuracy",acc_manual(test_out, Gx))
    print("ERMS VAL",mean_squared_error(val_out, Gx_val),"Accuracy",acc_manual(val_out, Gx_val))
#         erms.append(mean_squared_error(train_out[start:end], predict_train))
#         training_acc.append(acc_manual(train_out[start:end], predict_train))
 
def run_logistic_gsc_sub(train,test,val,train_out,test_out,val_out,BigSigma_inv):
    Weight_final = np.ones((CENTERS+1,),dtype=float64)

    erms = []
    for start in range(0,len(train),BATCH_GSC):
        end = start+BATCH_GSC
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

def run_logistic_gsc_con(train,test,val,train_out,test_out,val_out,BigSigma_inv):
    Weight_final = np.ones((CENTERS+1),dtype=float64)

    erms = []
    for start in range(0,len(train),BATCH_GSC):
        end = start+BATCH_GSC
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
def neural_net(train,test,val,train_out,test_out,val_out,BigSigma_inv):
    clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(100, 1), activation='logistic',batch_size=BATCH_HUMAN,shuffle=True,max_iter=5000)

    scaler = StandardScaler()
    scaler.fit(train)  
    train1 = scaler.transform(train)  
# apply same transformation to test data
    test = scaler.transform(test) 
    train_out = train_out.astype(float)
    clf.fit(X=train1, y=train_out)
    predict_test = clf.predict(test)
    predict_val = clf.predict(val)
    print("NN with HOD data : ")
    print("TEST ERMS ACCURACY",mean_squared_error(test_out, predict_test),acc_manual(test_out, predict_test))
    print("VAL ERMS ACCURACY",mean_squared_error(val_out, predict_val),acc_manual(val_out, predict_test))
def neural_net_2(train,test,val,train_out,test_out,val_out,BigSigma_inv):
    clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(100, 1), activation='logistic',batch_size=BATCH_HUMAN,shuffle=True,max_iter=5000)

    scaler = StandardScaler()
    scaler.fit(train)  
    train1 = scaler.transform(train)  
# apply same transformation to test data
    test = scaler.transform(test) 
    train_out = train_out.astype(float)
    clf.fit(X=train1, y=train_out)
    predict_test = clf.predict(test)
    predict_val = clf.predict(val)
    print("TEST ERMS ACCURACY",mean_squared_error(test_out, predict_test),acc_manual(test_out, predict_test))
    print("VAL ERMS ACCURACY",mean_squared_error(val_out, predict_val),acc_manual(val_out, predict_test))
def run_models():
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data(0)
#     linear_regress_human_sub(train,test,val,train_out,test_out,val_out,BigSigma_inv)
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data(1)
#     linear_regress_human_con(train,test,val,train_out,test_out,val_out,BigSigma_inv)
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data(0)
#     run_logistic_human_sub(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data(1)
#     run_logistic_human_con(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_gsc_data(0)
#     run_linear_reg_gsc_sub(train,test,val,train_out,test_out,val_out,BigSigma_inv)
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_gsc_data(1)
#     run_linear_reg_gsc_con(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_gsc_data(0)
#     run_logistic_gsc_sub(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_gsc_data(1)
#     run_logistic_gsc_con(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data(0)
#     neural_net(train, test, val, train_out, test_out, val_out, BigSigma_inv) 
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data(1)
#     neural_net(train, test, val, train_out, test_out, val_out, BigSigma_inv)
    train,test,val,train_out,test_out,val_out,BigSigma_inv = process_gsc_data(0)
    neural_net_2(train, test, val, train_out, test_out, val_out, BigSigma_inv)
    train,test,val,train_out,test_out,val_out,BigSigma_inv = process_gsc_data(1)
    neural_net_2(train, test, val, train_out, test_out, val_out, BigSigma_inv)

#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data(0)
#     linear_regress_human_sub(train,test,val,train_out,test_out,val_out,BigSigma_inv)
#     run_logistic_human_sub(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     neural_net(train, test, val, train_out, test_out, val_out, BigSigma_inv) 
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_human_data(1)
#     linear_regress_human_con(train,test,val,train_out,test_out,val_out,BigSigma_inv)
#     run_logistic_human_con(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     neural_net(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#  
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_gsc_data(0)
#     run_linear_reg_gsc_sub(train,test,val,train_out,test_out,val_out,BigSigma_inv)
#     run_logistic_gsc_sub(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     neural_net_2(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     train,test,val,train_out,test_out,val_out,BigSigma_inv = process_gsc_data(1)
#     run_linear_reg_gsc_con(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     run_logistic_gsc_con(train, test, val, train_out, test_out, val_out, BigSigma_inv)
#     neural_net_2(train, test, val, train_out, test_out, val_out, BigSigma_inv)
run_models()