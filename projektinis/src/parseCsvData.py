import csv
import sys, getopt

from normalizingData import normalizeMultipleDataLists

def multipleFileParse(data: list):
    finalList = []

    for name, fileName in data:
        rewardList = []

        with open(fileName, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            lineCounter = 0
            for row in spamreader:
                if lineCounter > 0:
                    rewardList.append(int(float(row[-1])))
                lineCounter += 1

        finalList.append((name, rewardList))

    print(finalList)
    normalizeMultipleDataLists(finalList, 500)

def main(argv):
    fileName = ''

    # Parse arguments
    try:
        opts, args = getopt.getopt(argv, 'f:', ['file='])
    except getopt.GetoptError:
        print ('--file <file-path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-f', '--file'):
            fileName = arg

    stepsList = []
    rewardList = []

    # Parse csv file (Format: Wall time, Step, Value)
    with open(fileName, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        lineCounter = 0
        for row in spamreader:
            if lineCounter > 0:
                stepsList.append(int(row[1]))
                rewardList.append(float(row[2]))

            lineCounter += 1

    print('Line count: ', lineCounter)

    normalizeMultipleDataLists(('test', stepsList, rewardList, 2e7))

if __name__ == '__main__':
    main(sys.argv[1:])