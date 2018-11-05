import numpy as np
import pandas as pd
from numpy import int64, float64, float32
from cmath import isnan, nan
from unittest.mock import inplace

data = pd.read_csv("GSC-Features.csv")
same = pd.read_csv("same_pairs_GSC.csv") 
diff = pd.read_csv("diffn_pairs_GSC.csv")



data_features = data.iloc[:,1:513]

print(data_features.shape)
data_features = data_features.astype(float64)
# print(data_features.isnull().any().any())
p = np.random.permutation(range(30000)) 
data_features = data_features.values
data = data.values
same = same.values
diff = diff.values
diff = diff[p]
same = same[p]
output=pd.DataFrame(same)
output = output.append(pd.DataFrame(diff),ignore_index=True)
output = output.values
print(output.shape)
subtract_features_inp = []

concat_input = []
 
def extract_gsc_data_sub():
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
         

    print("Raw input shape")
    print(len(subtract_features_inp))
    return subtract_features_inp
 
def extract_gsc_data_con():
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

    print(concat_input[0])
    return concat_input


