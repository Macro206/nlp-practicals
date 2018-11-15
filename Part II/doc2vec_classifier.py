from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize
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

def generateDoc2VecModel():
    trainingData = importData()

    print("Data imported")

    taggedDocuments = [TaggedDocument(doc, [i+1]) for i,doc in enumerate(trainingData)]
    model = Doc2Vec(taggedDocuments, dm=1, vector_size=100, max_epochs=10)

    model.save("/Users/Matteo/Desktop/doc2vec_models/model1")


def performDoc2VecClassification(trainingData, testData):
    doc2vecModel = Doc2Vec.load("/Users/Matteo/Desktop/doc2vec_models/model1")

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


generateDoc2VecModel()

# features = getFeaturesForAllReviews()
#
# trainingSplit = []
# testSplit = []
#
# trainingSplit.extend(features[0:900])
# trainingSplit.extend(features[1000:1900])
# testSplit.extend(features[900:1000])
# testSplit.extend(features[1900:2000])
#
# performDoc2VecClassification(features[0:1800], features[1800:2000])
