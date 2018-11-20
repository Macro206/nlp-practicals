from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize
from time import time
from review_loader import getFeaturesForAllReviews
import svmlight
import os
import string

### DOC2VEC MODEL TRAINING ###

def importFolder(folder):
    folderList = sorted(list(filter(lambda s: s.endswith(".txt"), os.listdir(folder))))

    reviews = []

    for fileName in folderList:
        with open(folder + fileName, 'r') as file:
            reviews.append(list(tokenize(file.read())))

    return reviews

def importData():
    # TODO: CHANGE THIS WHEN RUNNING ON ANOTHER MACHINE
    corpusRoot = "/Users/Matteo/Desktop/aclImdb/"

    paths = ["train/neg/",
             "train/pos/",
             "train/unsup/",
             "test/neg/",
             "test/pos/"]

    docs = []
    for p in paths:
        docs.extend(importFolder(corpusRoot+p))

    return docs

def importDataTokenized():
    trainingDataDirectory = '/Users/Matteo/Desktop/doc2vec_training_data/'

    folderList = os.listdir(trainingDataDirectory)

    reviews = []

    for fileName in folderList:
        with open(trainingDataDirectory + fileName, 'r') as file:
            reviews.append(filter(lambda x: x != '', file.read().split('\n')))

    return reviews


def generateDoc2VecModel():
    trainingData = importDataTokenized()

    print("Data imported")

    start_time = time()

    taggedDocuments = [TaggedDocument(doc, [i+1]) for i,doc in enumerate(trainingData)]
    model = Doc2Vec(taggedDocuments, dm=0, vector_size=100, epochs=10, window=5, hs=0, negative=20)

    model.save("/Users/Matteo/Desktop/doc2vec_models/sample")

    end_time = time()

    dt = int(end_time - start_time)

    print("Time taken: " + str(dt/60) + str(":") + str(dt%60))


def performDoc2VecClassification(trainingData, testData):
    doc2vecModel = Doc2Vec.load("/Users/Matteo/Desktop/doc2vec_models/final_model")

    trainingFeatureVectors = [(1 if doc[0] == 'POS' else -1, doc2vecModel.infer_vector(doc[2])) for doc in trainingData]
    testFeatureVectors = [(0, doc2vecModel.infer_vector(doc[2])) for doc in testData]

    formattedTrainingFeatureVectors = [(v[0], [(i+1,f) for i,f in enumerate(v[1])]) for v in trainingFeatureVectors]
    formattedTestFeatureVectors = [(v[0], [(i+1,f) for i,f in enumerate(v[1])]) for v in testFeatureVectors]

    svmModel = svmlight.learn(formattedTrainingFeatureVectors)
    judgements = svmlight.classify(svmModel, formattedTestFeatureVectors)

    predictions = []

    i = 0
    for (sentiment, fileName, features) in testData:
        judgement = "POS" if judgements[i] > 0 else "NEG"

        predictions.append((judgement, sentiment, fileName))

        i += 1

    return predictions


def performDoc2VecJudgement(trainingData, testData):
    doc2vecModel = Doc2Vec.load("/Users/Matteo/Desktop/doc2vec_models/final_model")

    trainingFeatureVectors = [(1 if doc[0] == 'POS' else -1, doc2vecModel.infer_vector(doc[2])) for doc in trainingData]
    testFeatureVectors = [(0, doc2vecModel.infer_vector(doc[2])) for doc in testData]

    formattedTrainingFeatureVectors = [(v[0], [(i+1,f) for i,f in enumerate(v[1])]) for v in trainingFeatureVectors]
    formattedTestFeatureVectors = [(v[0], [(i+1,f) for i,f in enumerate(v[1])]) for v in testFeatureVectors]

    svmModel = svmlight.learn(formattedTrainingFeatureVectors)
    judgements = svmlight.classify(svmModel, formattedTestFeatureVectors)

    predictions = []

    i = 0
    for (sentiment, fileName, features) in testData:
        predictions.append((judgements[i], sentiment, fileName))

        i += 1

    return predictions

# generateDoc2VecModel()
