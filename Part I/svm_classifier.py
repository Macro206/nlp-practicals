import svmlight
import naive_bayes

def getFeatureIndices(data):
    featureIndices = {}
    index = 1

    for doc in data:
        for f in doc[2]:
            if f not in featureIndices:
                featureIndices[f] = index
                index += 1

    return featureIndices


def performSVMClassification(trainingData, testData):
    featureIndices = getFeatureIndices(trainingData + testData)

    formattedTrainingData = []
    formattedTestData = []

    for doc in trainingData:
        featureVector = []
        features = {}

        for f in doc[2]:
            #features[f] = 1 if f not in features else features[f] + 1
            features[f] = 1

        for k,v in features.items():
            featureVector.append((featureIndices[k],v))

        list.sort(featureVector, key=lambda x: x[0])

        sentimentVal = 1 if doc[0] == "POS" else -1

        formattedTrainingData.append((sentimentVal, featureVector))

    model = svmlight.learn(formattedTrainingData)


    for doc in testData:
        featureVector = []
        features = {}

        for f in doc[2]:
            features[f] = 1 if f not in features else features[f] + 1

        for k,v in features.items():
            featureVector.append((featureIndices[k],v))

        list.sort(featureVector, key=lambda x: x[0])

        formattedTestData.append((0, featureVector))

    judgements = svmlight.classify(model, formattedTestData)

    formattedJudgements = []

    i = 0
    for (sentiment, fileName, features) in testData:
        formattedJudgements.append(("POS" if judgements[i] > 0 else "NEG", sentiment, fileName))
        i += 1

    return formattedJudgements
