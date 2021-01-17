import numpy as np
import matplotlib.pyplot as plt

def getMovingAverageData(data: list, avgSize = 10):
    newData = []
    i = 0

    while i < len(data) - avgSize + 1:
        listForAvg = data[i : i + avgSize]
        currentAvg = sum(listForAvg) / avgSize
        newData.append(currentAvg)
        i += 1

    return newData

def getDifferenceData(data: list):
    differenceData = []
    i = 1

    while i < len(data):
        difference = data[i] - data[i-1]
        differenceData.append(difference)
        i += 1

    return differenceData

def getSmoothData(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def normalizeMultipleDataLists(data, xAxisLabel = '', yAxisLabel = '', title = ''):

    plt.figure(1)
    plt.clf()

    for algName, xData, yData, _ in data:
        yDataSmooth = getSmoothData(yData, 0.999)
        plt.plot(xData, yDataSmooth, label=algName)

    plt.xlabel(xAxisLabel)
    plt.ylabel(yAxisLabel)
    plt.title(title)
    plt.legend()
    plt.pause(0.001)

    plt.show()
    
def polyfitMultipleDataLists(data, xAxisLabel, yAxisLabel, title):
    
    plt.figure(1)
    plt.clf()

    for algName, xData, yData, limit in data:
        yDataSmooth = getSmoothData(yData, 0.999)

        if limit == 2e7:
            value = xData[-1]
        else:
            value = next(x for x in xData if x > limit)
        index = xData.index(value)

        funct = np.polyfit(xData[:index], yDataSmooth[:index], 1)
        x = np.linspace(0, limit, 10000)
        y = funct[0] * x + funct[1] 
        print('AlgName: {}, Func: {}x + {}'.format(algName, funct[0], funct[1]))
        plt.plot(x,y, label=algName)

    plt.xlabel(xAxisLabel)
    plt.ylabel(yAxisLabel)
    plt.title(title)
    plt.legend()
    plt.pause(0.001)

    plt.show()

