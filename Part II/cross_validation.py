import naive_bayes
import svm_classifier
import permutation_test
from review_loader import *
import statistics

folds = 10

### SPLITTING TYPES ###

def roundRobinSplitting(features):
    positiveFeatures = features[0:1000]
    negativeFeatures = features[1000:2000]

    splits = []

    for i in range(0,folds):
        splits.append([])

    for i in range(0,1000):
        splitIndex = i % folds
        splits[splitIndex].append(positiveFeatures[i])
        splits[splitIndex].append(negativeFeatures[i])

    return splits


### CROSS VALIDATION FOR SPECIFIC SYSTEM ###

def crossValidateForSpecificIndex(splits, testingSplitIndex, func):
    trainingData = []
    testData = []

    for i in range(0,len(splits)):
        if i == testingSplitIndex:
            testData = splits[i]
        else:
            trainingData.extend(splits[i])

    if options["shouldUseCutoffs"]:
        featureFrequencies = {}
        for (sentiment, fileName, features) in trainingData:
            for f in features:
                featureFrequencies[f] = 1 if f not in featureFrequencies else featureFrequencies[f] + 1

        for i in range(0,len(trainingData)):
            newFeatures = []
            for f in trainingData[i][2]:
                if featureFrequencies[f] >= 4:
                    newFeatures.append(f)

            trainingData[i] = (trainingData[i][0], trainingData[i][1], newFeatures)

    classificationResults = func(trainingData, testData)

    totalCorrect = 0

    for i in range(0,len(classificationResults)):
        if classificationResults[i][0] == classificationResults[i][1]:
            totalCorrect += 1


    return float(totalCorrect)/float(len(classificationResults))


def crossValidate(splits, func):
    scores = []

    for i in range(0,len(splits)):
        scores.append(crossValidateForSpecificIndex(splits, i, func))

    return scores


def crossValidateNaiveBayes():
    features = getFeaturesForAllReviews()

    roundRobinSplits = roundRobinSplitting(features)

    roundRobinScores = crossValidate(roundRobinSplits, naive_bayes.naiveBayes)

    print "Naive Bayes:"
    print "Mean: " + str(statistics.mean(roundRobinScores))


def crossValidateSVM():
    features = getFeaturesForAllReviews()

    roundRobinSplits = roundRobinSplitting(features)

    roundRobinScores = crossValidate(roundRobinSplits, svm_classifier.performSVMClassification)

    print "SVM:"
    print "Mean: " + str(statistics.mean(roundRobinScores))


### CROSS VALIDATION COMPARING SYSTEMS ###

def getCrossValidatedJudgementsForSpecificIndex(splits, testingSplitIndex, func):
    trainingData = []
    testData = []

    for i in range(0,len(splits)):
        if i == testingSplitIndex:
            testData = splits[i]
        else:
            trainingData.extend(splits[i])

    if options["shouldUseCutoffs"]:
        featureFrequencies = {}
        for (sentiment, fileName, features) in trainingData:
            for f in features:
                featureFrequencies[f] = 1 if f not in featureFrequencies else featureFrequencies[f] + 1

        for i in range(0,len(trainingData)):
            newFeatures = []
            for f in trainingData[i][2]:
                if featureFrequencies[f] >= 4:
                    newFeatures.append(f)

            trainingData[i] = (trainingData[i][0], trainingData[i][1], newFeatures)

    classificationResults = func(trainingData, testData)

    return classificationResults

def getCrossValidatedJudgements(splits, func):
    results = []

    for i in range(0,len(splits)):
        results.extend(getCrossValidatedJudgementsForSpecificIndex(splits, i, func))

    return results

def getAggregatedJudgementsForSystem(func):
    features = getFeaturesForAllReviews()
    roundRobinSplits = roundRobinSplitting(features)
    judgements = getCrossValidatedJudgements(roundRobinSplits, func)
    return judgements

def compareSystems(func1, func2):
    judgements1 = getAggregatedJudgementsForSystem(func1)
    judgements2 = getAggregatedJudgementsForSystem(func2)

    pValue = permutation_test.compareResults(judgements1, judgements2)

    print "Comparing systems"
    print "Got p-value: " + str(pValue)


print "Options: "
print options
print ""

#crossValidateNaiveBayes()

#print ""
#print "-------"
#print ""

#crossValidateSVM()

#print ""
#print "-------"
#print ""

compareSystems(naive_bayes.naiveBayes, naive_bayes.naiveBayes)
