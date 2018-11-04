from math import log

### FUNCTIONS TO CALCULATE PROBABILITIES ###

def calculateLogProbs(trainingSet):
    frequencies = {}
    logProbs = {}

    totalPositiveWords = 0
    totalNegativeWords = 0

    for (sentiment, fileName, features) in trainingSet:
        isPositive = (sentiment == "POS")

        for s in features:
            if isPositive:
                totalPositiveWords += 1
            else:
                totalNegativeWords += 1

            if s not in frequencies:
                frequencies[s] = (1,1)  # Add one smoothing

            (currentPositiveFreq, currentNegativeFreq) = frequencies[s]

            if isPositive:
                frequencies[s] = (currentPositiveFreq + 1, currentNegativeFreq)
            else:
                frequencies[s] = (currentPositiveFreq, currentNegativeFreq + 1)

    # More add one smoothing
    totalPositiveWords += len(frequencies)
    totalNegativeWords += len(frequencies)

    for s in frequencies:
        (positiveFreq, negativeFreq) = frequencies[s]

        wordProbPositive = float(positiveFreq) / float(totalPositiveWords)
        wordProbNegative = float(negativeFreq) / float(totalNegativeWords)

        logProbs[s] = (log(wordProbPositive), log(wordProbNegative))

    return logProbs


def calculateClassProbabilities(trainingSet):
    positiveCount = 0
    negativeCount = 0

    for (sentiment, fileName, features) in trainingSet:
        if sentiment == "POS":
            positiveCount += 1
        else:
            negativeCount += 1

    return (float(positiveCount)/float(positiveCount+negativeCount), float(negativeCount)/float(positiveCount+negativeCount))

### FUNCTION TO PERFORM CLASSIFICATION ###

def naiveBayes(trainingSet, testSet):
    featureLogProbs = calculateLogProbs(trainingSet)
    classProbs = calculateClassProbabilities(trainingSet)

    predictions = []

    for (sentiment, fileName, features) in testSet:
        positiveProb = log(classProbs[0])
        negativeProb = log(classProbs[1])

        for s in features:
            if s not in featureLogProbs:
                continue

            (positiveWordProb, negativeWordProb) = featureLogProbs[s]

            positiveProb += positiveWordProb
            negativeProb += negativeWordProb

        if (positiveProb >= negativeProb):
            predictions.append(("POS", sentiment, fileName))
        else:
            predictions.append(("NEG", sentiment, fileName))

    return predictions
