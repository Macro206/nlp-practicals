import random

R = 5000

def getMeanForResults(results):
    total = 0
    correct = 0

    for r in results:
        if r[0] == r[1]:
            correct += 1
        total += 1

    return float(correct)/float(total)

def generateRandomSwapConfig(size):
    return [random.randint(0,1) for _ in range(0,size)]

def getDiffForSwapConfig(s1results, s2results, config):
    newS1Results = [s1results[i] if val == 0 else s2results[i] for i,val in enumerate(config)]
    newS2Results = [s2results[i] if val == 0 else s1results[i] for i,val in enumerate(config)]

    mean1 = getMeanForResults(newS1Results)
    mean2 = getMeanForResults(newS2Results)

    return abs(mean1 - mean2)


def compareResults(s1results, s2results):
    list.sort(s1results, key=lambda x: x[2])
    list.sort(s2results, key=lambda x: x[2])

    if len(s1results) != len(s2results):
        print("Unequal numbers of judgements for permutation test")
        quit()

    nJudgements = len(s1results)

    for i in range(0,nJudgements):
        if s1results[i][2] != s2results[i][2]:
            print("Files not equal: " + s1results[i][2] + " and " + s2results[i][2])
            return

    baselineDiff = getDiffForSwapConfig(s1results, s2results, [0 for _ in range(0,nJudgements)])

    randomSwapConfigs = [generateRandomSwapConfig(nJudgements) for _ in range(0,R)]
    diffs = [getDiffForSwapConfig(s1results, s2results, config) for config in randomSwapConfigs]

    s = sum(1 if diff >= baselineDiff else 0 for diff in diffs)

    pValue = float(s + 1)/float(R + 1)

    return pValue
