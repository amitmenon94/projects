import numpy as np

def import_data(file_path):
    data_target = []
    data_input = []
    with open(file_path, 'r') as f:
        while True:
            # Splitting our dataset per colums based on the space separator
            data_str = f.readline().split(' ')
            if not data_str[0]:
                break
            # Target is the first column
            data_target.append(float(data_str[0]))
            # Features are columns 2 to 48, where we take only the values
            data_input.append([float(a.split(':')[1]) for a in data_str[2:48]])

    x = np.array(data_input)
    y = np.array(data_target).reshape((-1, 1))

    return x, y

file_path = r'C:\Users\a\Querylevelnorm.txt'
x, y = import_data(file_path)
print("X shape is", x.shape, "\nY shape is", y.shape)