import naive_bayes
import svm_classifier
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


### CROSS VALIDATION FOR NAIVE BAYES ###

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
    features = naive_bayes.getFeaturesForAllReviews()

    consecutiveSplits = consecutiveSplitting(features)
    roundRobinSplits = roundRobinSplitting(features)

    consecutiveScores = crossValidate(consecutiveSplits, naive_bayes.naiveBayes)
    roundRobinScores = crossValidate(roundRobinSplits, naive_bayes.naiveBayes)

    print "Consecutive splitting:"
    print "Mean: " + str(statistics.mean(consecutiveScores))
    print "Variance: " + str(statistics.variance(consecutiveScores))

    print ""

    print "Round-robin splitting:"
    print "Mean: " + str(statistics.mean(roundRobinScores))
    print "Variance: " + str(statistics.variance(roundRobinScores))


def crossValidateSVM():
    features = naive_bayes.getFeaturesForAllReviews()

    roundRobinSplits = roundRobinSplitting(features)

    roundRobinScores = crossValidate(roundRobinSplits, svm_classifier.performSVMClassification)

    print "Round-robin splitting:"
    print "Mean: " + str(statistics.mean(roundRobinScores))
    print "Variance: " + str(statistics.variance(roundRobinScores))



crossValidateSVM()
