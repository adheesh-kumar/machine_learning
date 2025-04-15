import numpy as np
import matplotlib.pyplot as plt
import math

##O(1): bubbleSort is O(n^2) but since only used with A of len <= 5, O(25) -> O(1)
def medianOfFive(AO):
    A = AO.copy()
    n = len(A)

    for i in range(n):
        for j in range(n):
            if (A[i] > A[j]):
                tmp = A[i]
                A[i] = A[j]
                A[j] = tmp

    

    return A[n//2]

##medians of medians x=5
def medianOfMedians(A, indices, k):

    medians = []
    remainder = len(A) % 5

   
    isOdd = 0
    
    if (remainder > 0):
        isOdd = 1

    newIndices = []
    for i in range((len(A) // 5) + isOdd):
        medians.append(medianOfFive(A[i*5:i*5 + 5]))
        newIndices.append(i)

    if (len(medians) <= 5):
        pivot = medianOfFive(medians)
    else:
        pivot = medianOfMedians(medians, newIndices, len(medians)//2)


    low = []
    high = []
    equal = []

    indicesL = []
    indicesH = []
    indicesE = []

    for i in range (len(A)):
        if (A[i] > pivot):
            high.append(A[i])
            indicesH.append(indices[i])
        elif (A[i] < pivot):
            low.append(A[i])
            indicesL.append(indices[i])
        else:
            equal.append(A[i])
            indicesE.append(indices[i])

    i = 0

    for x in range(len(low)):
        A[i] = low[x]
        indices[i] = indicesL[x]
        i+=1

    for x in range(len(equal)):
        A[i] = equal[x]
        indices[i] = indicesE[x]
        i+=1
    
    for x in range(len(high)):
        A[i] = high[x]
        indices[i] = indicesH[x]
        i+=1


    K = len(low)

    if (K == k): 
        return pivot
    elif (K < k):
        return medianOfMedians(high, indicesH, k - K - 1)
    else:
        return medianOfMedians(low, indicesL, k)






def kNN(X, Y, x, k):

    dist = []
    indices = []
    dim = len(X[0])
    for i in range(len(X)):
        indices.append(i)
        sum = 0
        for j in range(dim):
            sum += (X[i][j] - x[j]) * (X[i][j] - x[j])
        dist.append(math.sqrt(sum))

    medianOfMedians(dist, indices, k)

    sum = 0
    for i in range(k):
        sum += Y[indices[i]][0]

    return sum / k




def readCSV(filename):
    X = []
    file = open(filename)
    
    for line in file:
        line = line.rstrip('\n')
        x = line.split(',')
        X.append([float(i) for i in x])

    file.close()

    return X

X = readCSV("X_train_D.csv")
Y = readCSV("Y_train_D.csv")

testX = readCSV("X_test_D.csv")
testY = readCSV("Y_test_D.csv")

errors = []

for k in range(9):
    sum = 0
    for i in range(len(testX)):
        guessY = kNN(X, Y, testX[i], k + 1)

        sum += math.sqrt((testY[i][0] - guessY) * (testY[i][0] - guessY))

    errors.append(sum / len(testX))

print(errors)


plt.plot(errors)
plt.show()













