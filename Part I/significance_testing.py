import math
from decimal import *

def factorial(n):
    acc = 1
    while n > 0:
        acc *= n
        n -= 1

    return acc


def getPValue(plus, minus, null):
    getcontext().prec = 10000

    print str(null) + " : " + str(plus) + " : " + str(minus)

    q = 0.5
    N = int(2*(math.ceil(float(null)/2.0)) + plus + minus)
    k = int(math.ceil(float(null)/2.0) + min(plus, minus))

    acc = Decimal(0)

    for i in range(0,k+1):
        #print i

        nCr = factorial(N)/(factorial(i)*factorial(N-i))

        acc += Decimal(nCr) * Decimal(q ** i) * Decimal((1-q) ** (N-i))

    return 2*float(acc)


def compareResults(s1results, s2results):
    list.sort(s1results, key=lambda x: x[2])
    list.sort(s2results, key=lambda x: x[2])

    plus = 0
    minus = 0
    null = 0

    oneSuccesses = 0
    twoSuccesses = 0
    total = 0

    for i in range(0,len(s1results)):
        (predictedSentiment1, actualSentiment1, file1) = s1results[i]
        (predictedSentiment2, actualSentiment2, file2) = s2results[i]

        if file1 != file2:
            print "Files not equal: " + file1 + " and " + file2
            return

        if (predictedSentiment1 == actualSentiment1) and (predictedSentiment2 != actualSentiment2):
            plus += 1
            oneSuccesses += 1
        elif (predictedSentiment1 != actualSentiment1) and (predictedSentiment2 == actualSentiment2):
            minus += 1
            twoSuccesses += 1
        else:
            if (predictedSentiment1 == actualSentiment1):
                oneSuccesses += 1
                twoSuccesses += 1
            null += 1

        total += 1

    print "System 1 Accuracy:"
    print float(oneSuccesses)/float(total)
    print ""

    print "System 2 Accuracy:"
    print float(twoSuccesses)/float(total)
    print ""

    return getPValue(plus, minus, null)
