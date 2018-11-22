import naive_bayes
import svm_classifier
import doc2vec_classifier
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

    return splits[1:]  # We discard the validation corpus


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


def crossValidateDoc2Vec():
    features = getFeaturesForAllReviews()

    roundRobinSplits = roundRobinSplitting(features)

    roundRobinScores = crossValidate(roundRobinSplits, doc2vec_classifier.performDoc2VecClassification)

    print "Doc2Vec:"
    print "Mean: " + str(statistics.mean(roundRobinScores))


def tuneParamsForDoc2Vec():
    features = getFeaturesForAllReviews()

    roundRobinSplits = roundRobinSplitting(features)

    validationCorpus = roundRobinSplits[0]
    trainingCorpus = [review for split in roundRobinSplits[1:] for review in split]

    classificationResults = doc2vec_classifier.performDoc2VecClassification(trainingCorpus, validationCorpus)

    totalCorrect = 0

    for i in range(0,len(classificationResults)):
        if classificationResults[i][0] == classificationResults[i][1]:
            totalCorrect += 1

    print "Doc2Vec:"
    print "Score: " + str(float(totalCorrect)/float(len(classificationResults)))


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
        print("Index " + str(i) + " concluded")

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


### DOC2VEC INVESTIGATION ###

def judgementIsCorrect(j):
    return (j[0] > 0 and j[1] == 'POS') or (j[0] <= 0 and j[1] == 'NEG')

def saveDoc2VecResults():
    judgements = getAggregatedJudgementsForSystem(doc2vec_classifier.performDoc2VecJudgement)

    with open('./doc2vec_judgements.txt', 'w') as f:
        for (judgement, actualSentiment, fileName) in judgements:
            f.write(str(judgement) + ',' + actualSentiment + ',' + fileName + '\n')

def loadDoc2VecResults():
    with open('./doc2vec_judgements.txt', 'r') as f:
        lines = f.read().split('\n')

        judgements = []

        for l in lines:
            if l == '':
                continue

            components = l.split(',')
            judgements.append((float(components[0]), components[1], components[2]))

    return judgements

def findMostConfidentlyIncorrect(judgements):
    judgement = None
    highestConfidence = 0

    for (j, s, f) in judgements:
        if not judgementIsCorrect((j,s,f)):
            if abs(j) > highestConfidence:
                judgement = (j,s,f)
                highestConfidence = abs(j)

    print(judgement)

def testFuncOnAmazonDataset(func, dataset):
    amazonDocuments = getFeaturesForAllAmazonReviews(dataset)
    imdbTrainingSplits = roundRobinSplitting(getFeaturesForAllReviews())
    imdbTrainingDocuments = [review for split in imdbTrainingSplits for review in split]

    classificationResults = func(imdbTrainingDocuments, amazonDocuments)

    totalCorrect = 0

    for i in range(0,len(classificationResults)):
        if classificationResults[i][0] == classificationResults[i][1]:
            totalCorrect += 1

    print "Amazon dataset:"
    print "Score: " + str(float(totalCorrect)/float(len(classificationResults)))

    return classificationResults

def compareSystemsOnAmazonDataset(func1, func2, dataset):
    judgements1 = testFuncOnAmazonDataset(func1, dataset)
    options['shouldUsePresence'] = True
    judgements2 = testFuncOnAmazonDataset(func2, dataset)

    pValue = permutation_test.compareResults(judgements1, judgements2)

    print "Comparing systems"
    print "Got p-value: " + str(pValue)

def compareSystemsOnAmazonDatasets(datasets_to_test):
    for dataset in datasets_to_test:
        print("")
        print("Testing Naive Bayes on dataset: " + dataset)
        testFuncOnAmazonDataset(naive_bayes.naiveBayes, dataset)

    print("")
    print("-----")
    print("")

    options['shouldUsePresence'] = True

    for dataset in datasets_to_test:
        print("")
        print("Testing SVM with presence on dataset: " + dataset)
        testFuncOnAmazonDataset(svm_classifier.performSVMClassification, dataset)

    options['shouldUsePresence'] = False

    print("")
    print("-----")
    print("")

    for dataset in datasets_to_test:
        print("")
        print("Testing Doc2Vec on dataset: " + dataset)
        testFuncOnAmazonDataset(doc2vec_classifier.performDoc2VecClassification, dataset)


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

#compareSystems(svm_classifier.performSVMClassification, doc2vec_classifier.performDoc2VecClassification)
#crossValidateDoc2Vec()

#saveDoc2VecResults()

# judgements = loadDoc2VecResults()
#
# totalCorrect = 0
# for j in judgements:
#     if judgementIsCorrect(j):
#         totalCorrect += 1
#
# print(float(totalCorrect)/float(len(judgements)))
#
# findMostConfidentlyIncorrect(judgements)

#compareSystemsOnAmazonDatasets(amazon_datasets)

compareSystemsOnAmazonDataset(naive_bayes.naiveBayes, svm_classifier.performSVMClassification, "video_games")
