# AUTHOR: JOHN DOGAN
'''
1. PROGRAM COLLECTS FEATURES
2. PROGRAM TESTS USING K-NEAREST-NEIGHBOR
3. PROGRAM PREPARES CALCULATIONS
'''


import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def main():

    #STORES ALL FEATURES for Training
    #10 SAMPLES IN TOTAL
    samplesTraining = {}

    #STORES ALL FEATURES for Testing
    #10 SAMPLES IN TOTAL
    samplesTesting = {}

    #1. COLLECT FEATURES FROM FILES
    # finTrain is the Training Set
    # finTest is the Testing Set
    for i in range(10):

        # IF COLLECTING MALES
        if(i <= 4):
            finTrain = open("Face Database/m-00" + str(i+1) + "/" + "m-00" + str(i+1) + "-01.pts")

            #RETRIEVE FEATURES FROM SAMPLE
            samplesTraining[i+1] = collect(finTrain)

            finTest = open("Face Database/m-00" + str(i + 1) + "/" + "m-00" + str(i + 1) + "-05.pts")

            # RETRIEVE FEATURES FROM SAMPLE
            samplesTesting[i + 1] = collect(finTest)

        # COLLECTING FROM WOMEN
        else:
            finTrain = open("Face Database/w-00" + str((i-5)+1) + "/" + "w-00" + str((i-5)+1) + "-01.pts")

            # RETRIEVE FEATURES FROM SAMPLE
            samplesTraining[i+1] = collect(finTrain)

            finTest = open("Face Database/w-00" + str((i - 5) + 1) + "/" + "w-00" + str((i - 5) + 1) + "-05.pts")

            # RETRIEVE FEATURES FROM SAMPLE
            samplesTesting[i + 1] = collect(finTest)

    #2. TEST SAMPLES USING K-NEAREST-NEIGHBOR
    y_true, y_pred = test(samplesTraining, samplesTesting)

    #3. CHECK DATA USING CONFUSION MATRIX
    check(y_true, y_pred)

'''
Collect 7 features from sample file
1. Eye length ratio
2. Eye distance ratio
3. Nose Ratio
4. Lip Size Ratio
5. Lip length Ratio
6. Eye-Brow length ratio
7. Aggresive ratio
@features: return features which is a list containing 7 features for the sample
'''
def collect(file):

    #STORES ALL FEATURES TO SEND BACK
    features = []

    #CREATE LIST FROM FILE
    points = file.readlines()
    #LINES WE NEED TO CONSIDER
    points = points[3:-1]

    #GET X AND Ys seperate
    #ORDERED FROM 1 - 22 (Starts at 0)
    X = []
    Y = []

    #ORGANIZES X AND Y FOR POINTS
    for point in points:

        line = point.split(sep = " ")

        X.append(float(line[0]))

        Y.append(float(line[1][:-1]))

    #1. LENGTH OF EYE OVER DIST BETWEEN POINT 8 AND 13
    leftEye = math.sqrt((X[9] - X[10])**2 + (Y[9] - Y[10])**2)
    rightEye = math.sqrt((X[11] - X[12])**2 + (Y[11] - Y[12])**2)

    #LARGEST EYE IS DIVIDED BY DISTANCE
    if(leftEye >= rightEye):
        eyeLengthFeature = leftEye / math.sqrt((X[8] - X[13])**2 + (Y[8] - Y[13])**2)
    else:
        eyeLengthFeature = rightEye / math.sqrt((X[8] - X[13])**2 + (Y[8] - Y[13])**2)

    #2. DISTANCE BETWEEN CENTER OF TWO EYES OVER DISTANCE BETWEEN POINTS 8 AND 13
    eyeDistanceFeature = math.sqrt((X[0] - X[1])**2 + (Y[0] - Y[1])**2) \
                         / math.sqrt((X[8] - X[13])**2 + (Y[8] - Y[13])**2)

    #3. DISTANCE BETWEEN 15 and 16 OVER distance between 20 and 21
    noseRatio = math.sqrt((X[15] - X[16])**2 + (Y[15] - Y[16])**2) \
                         / math.sqrt((X[20] - X[21])**2 + (Y[20] - Y[21])**2)

    #4. DISTANCE BETWEEN POINTS 2 and 3 OVER distance 17 and 18
    lipSize = math.sqrt((X[2] - X[3])**2 + (Y[2] - Y[3])**2) \
                         / math.sqrt((X[17] - X[18])**2 + (Y[17] - Y[18])**2)

    #5. DISTANCE BETWEEN POINTS 2 and 3 over distance between 20 and 21
    lipLengthFeature = math.sqrt((X[2] - X[3])**2 + (Y[2] - Y[3])**2) \
                         / math.sqrt((X[20] - X[21])**2 + (Y[20] - Y[21])**2)

    #6. DISTANCE BETWEEN POINTS 4 and 5 over distance between 8 and 13
    browLengthFeature = math.sqrt((X[4] - X[5])**2 + (Y[4] - Y[5])**2) \
                         / math.sqrt((X[8] - X[13])**2 + (Y[8] - Y[13])**2)

    #7. DISTANCE BETWEEN POINTS 10 and 19 over distance between 20 and 21
    aggresiveRatio = math.sqrt((X[10] - X[19])**2 + (Y[10] - Y[19])**2) \
                         / math.sqrt((X[20] - X[21])**2 + (Y[20] - Y[21])**2)

    #PREPARE FEATURES TO SEND BACK
    features.append(eyeLengthFeature)
    features.append(eyeDistanceFeature)
    features.append(noseRatio)
    features.append(lipSize)
    features.append(lipLengthFeature)
    features.append(browLengthFeature)
    features.append(aggresiveRatio)

    return features;

'''
Test data using K-Nearest-Neighbor
@Y: returns true classes from training data
@predicted: returns predicted classes from test data
'''
def test(train,test):

    #FEATURES
    X = []
    # 10 CLASSES
    Y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    #TESTED DATA AND CLASS IT BELONGS TO
    predicted = []

    #TRAINING DATA
    #CREATE LIST FROM EACH SAMPLE FEATURE
    for i in range(10):

        X.append(list(train[i+1]))

    #K NEIGHBOR GATHER DATA
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X, Y)

    #CREATE LIST OF CLASSES FROM TEST DATA
    for i in range(10):

        predicted.append(int(neigh.predict(test[i+1])))

    #RETURNS TRUE AND PREDICTED DATA
    return Y, predicted;

'''
Check data using confusion matrix then write to file
'''
def check(y_true, y_pred):

    win = open('data.txt', 'w')

    #Confusion Matrix Table
    result = confusion_matrix(y_true, y_pred)

    #Number of classes in order
    target_names = ['P1', 'P2', 'P3','P4','P5','P6','P7','P8','P9','P10']

    #ACCURACY
    accuracy = accuracy_score(y_true,y_pred)

    #WRITE PREDICTED DATA
    win.write("PREDICTIONS FROM TEST DATA: " + str(y_pred) + "\n\n")
    #WRITE ACCURACY
    win.write("ACCURACY: " + str(accuracy) + "\n\n")
    #WRITE CONFUSION MATRIX
    win.write("CONFUSION MATRIX:\n")
    win.write(str(result) + '\n\n\n')
    win.write(str(classification_report(y_true, y_pred, target_names=target_names)))

    #FILE CLOSED
    win.close()

main()