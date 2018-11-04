import naive_bayes
import svm_classifier
import significance_testing
import review_loader
import statistics

### SPLITTING TYPES ###

def consecutiveSplitting(features):
    positiveFeatures = features[0:1000]
    negativeFeatures = features[1000:2000]

    splits = []

    for i in range(0,10):
        splits.append([])

    for i in range(0,1000):
        splitIndex = int(i/100)
        splits[splitIndex].append(positiveFeatures[i])
        splits[splitIndex].append(negativeFeatures[i])

    return splits

def roundRobinSplitting(features):
    positiveFeatures = features[0:1000]
    negativeFeatures = features[1000:2000]

    splits = []

    for i in range(0,10):
        splits.append([])

    for i in range(0,1000):
        splitIndex = i % 10
        splits[splitIndex].append(positiveFeatures[i])
        splits[splitIndex].append(negativeFeatures[i])

    return splits


### CROSS VALIDATION FOR SPECIFIC SYSTEM ###

def crossValidateForSpecificIndex(splits, testingSplitIndex, func):
    trainingData = []
    testData = []

    for i in range(0,10):
        if i == testingSplitIndex:
            testData = splits[i]
        else:
            trainingData.extend(splits[i])

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
    features = review_loader.getFeaturesForAllReviews()

    roundRobinSplits = roundRobinSplitting(features)

    roundRobinScores = crossValidate(roundRobinSplits, naive_bayes.naiveBayes)

    print "Naive Bayes:"
    print "Mean: " + str(statistics.mean(roundRobinScores))


def crossValidateSVM():
    features = review_loader.getFeaturesForAllReviews()

    roundRobinSplits = roundRobinSplitting(features)

    roundRobinScores = crossValidate(roundRobinSplits, svm_classifier.performSVMClassification)

    print "SVM:"
    print "Mean: " + str(statistics.mean(roundRobinScores))


### CROSS VALIDATION COMPARING SYSTEMS ###

def runSystemsForSpecificIndex(splits, testingSplitIndex, func1, func2):
    trainingData = []
    testData = []

    for i in range(0,10):
        if i == testingSplitIndex:
            testData = splits[i]
        else:
            trainingData.extend(splits[i])

    classificationResults1 = func1(trainingData, testData)
    classificationResults2 = func2(trainingData, testData)

    return (classificationResults1, classificationResults2)


def compareSystems(func1, func2):
    s1results = [[] for _ in range(0,10)]
    s2results = [[] for _ in range(0,10)]

    features = review_loader.getFeaturesForAllReviews()
    roundRobinSplits = roundRobinSplitting(features)

    for i in range(0,len(roundRobinSplits)):
        (results1, results2) = runSystemsForSpecificIndex(roundRobinSplits, i, func1, func2)

        s1results[i] = results1
        s2results[i] = results2

    pValues = significance_testing.compareSystems(s1results, s2results)

    print "Comparing systems"
    print "Got p-values: " + str(pValues)


print ""

crossValidateNaiveBayes()

print ""
print "-------"
print ""

crossValidateSVM()

print ""
print "-------"
print ""

#compareSystems(naive_bayes.naiveBayes, svm_classifier.performSVMClassification)
