import numpy as np
import pandas as pd
from sklearn.model_selection._split import train_test_split
from sklearn.neural_network import MLPClassifier    
import math
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster.k_means_ import KMeans
from sklearn.metrics.regression import mean_squared_error
from ExtractHumanData import extract_human_data_sub, extract_human_data_con,\
    output


target = output[:,2]




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

def process_gsc_data(value):
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



NUM_HIDDEN_NEURONS_LAYER_1 = 100
LEARNING_RATE = 0.08

train, test,val,train_out,test_out,val_out,BigSigma_inv = process_gsc_data(0)#process_human_data(0)
NUM_OF_EPOCHS = 5000
BATCH_SIZE = 100



clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(100, 1), activation='logistic',batch_size=BATCH_SIZE,shuffle=True,max_iter=NUM_OF_EPOCHS)

scaler = StandardScaler()
scaler.fit(train)  
train1 = scaler.transform(train)  
# apply same transformation to test data
test = scaler.transform(test) 
train_out = train_out.astype(float)
clf.fit(X=train1, y=train_out)
predict_test = clf.predict(test)
print(mean_squared_error(test_out, predict_test))
