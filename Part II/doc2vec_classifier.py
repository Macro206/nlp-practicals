from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from review_loader import getFeaturesForAllReviews
import svmlight

def trainModel(trainingData):
    taggedDocuments = [TaggedDocument(doc[2], [doc[1]]) for i,doc in enumerate(trainingData)]
    model = Doc2Vec(taggedDocuments, dm=1, vector_size=100, max_epochs=10)

    return model

def performDoc2VecClassification(trainingData, testData):
    doc2vecModel = trainModel(trainingData)

    trainingFeatureVectors = [(1 if doc[0] == 'POS' else -1, doc2vecModel.infer_vector(doc[2])) for doc in trainingData]
    testFeatureVectors = [(0, doc2vecModel.infer_vector(doc[2])) for doc in testData]

    formattedTrainingFeatureVectors = [(v[0], [(i+1,f) for i,f in enumerate(v[1])]) for v in trainingFeatureVectors]
    formattedTestFeatureVectors = [(v[0], [(i+1,f) for i,f in enumerate(v[1])]) for v in testFeatureVectors]

    svmModel = svmlight.learn(formattedTrainingFeatureVectors)
    judgements = svmlight.classify(svmModel, formattedTestFeatureVectors)

    nCorrect = 0

    i = 0
    for (sentiment, fileName, features) in testData:
        judgement = "POS" if judgements[i] > 0 else "NEG"

        if judgement == sentiment:
            nCorrect += 1

        i += 1

    print(float(nCorrect)/float(len(judgements)))




features = getFeaturesForAllReviews()

trainingSplit = []
testSplit = []

trainingSplit.extend(features[0:900])
trainingSplit.extend(features[1000:1900])
testSplit.extend(features[900:1000])
testSplit.extend(features[1900:2000])

performDoc2VecClassification(features[0:1800], features[1800:2000])
