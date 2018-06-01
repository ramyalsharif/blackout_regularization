import itertools
import numpy as np

step = 5
start = 50
end = 95

results = []

def readFile(fileName):
    data = open(filename, 'r')
    lines = data.readlines();
    testVals = []
    for line in lines:
        line = line.strip('\n');
        splitLine = line.split(',');
        splitLine = splitLine[-2:]
        splitLine = list(map(float, splitLine))
        testVals.append(splitLine)
    testVals.sort(key = lambda x: x[0], reverse = True)
    return testVals[0:20]


for i in range(start ,end+step, step):
    filename = 'AccuracyResultsMNISTBlackoutblackout0.' + str(i).rstrip('0') + '.txt'
    results.append(readFile(filename))

filename = 'AccuracyResultsMNISTBlackoutblackout1.0.txt'
results.append(readFile(filename))

output = []
output.append(['BlackoutParam', 'ValE', 'ValSD', 'TestE', 'TestSD']);

for res in results:
    valE = []
    testE = []
    for val in res:
        valE.append(val[0])
        testE.append(val[1])
    meanValE = np.mean(valE)
    meanTestE = np.mean(testE)
    SDValE = np.std(valE)
    SDTestE = np.std(testE)
    output.append([start, meanValE, SDValE, meanTestE, SDTestE])
    start = start + 5

with open('ChangingBlackoutOutput.txt', 'w') as filehandle:  
    for listitem in output:
        for listitemitem in listitem[:-1]:
            filehandle.write('%s, ' % listitemitem)
        filehandle.write('%s\n' % listitem[-1])