import numpy as np


#ATAw=ATz

def ridgeReg(A, z, reg=0):
    At = np.transpose(A)

    X = At @ A

    I = np.identity(np.shape(X)[0])

    X += reg*I

    b = At @ z

    w = np.linalg.solve(X, b)
    return w



def readCSV(filename):
    X = []
    
    file = open(filename)
    
    for line in file:
        line = line.rstrip('\n')
        x = line.split(',')
        X.append([float(i) for i in x])

    file.close()

    A = np.swapaxes(np.array(X), 0, 1)

    return A

A = readCSV("housing_X_train.csv")
y = np.squeeze(readCSV("housing_Y_train.csv"))

print(A, np.shape(A))

print(y, np.shape(y))


print(np.mean(ridgeReg(A, y, 0)))