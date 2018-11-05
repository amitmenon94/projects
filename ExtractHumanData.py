import numpy as np
import pandas as pd
from numpy import int64, float64, float32
from cmath import isnan

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


p = np.random.permutation(range(791)) 

data = data.values
same = same.values
diff = diff.values
diff = diff[p]

output=pd.DataFrame(same)
output = output.append(pd.DataFrame(diff))
output = output.values
data_features = data[:,1:10]

data_features = data_features.astype(float64)



data_features = np.nan_to_num(data_features)

subtract_features_inp = []
absolute = []
concat_input = []



def extract_human_data_sub():
    print("Extacting data")
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
        
    print("Done")
    return subtract_features_inp

def extract_human_data_con():
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

    print(concat_input[0])
    return concat_input


