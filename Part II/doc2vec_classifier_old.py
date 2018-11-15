import numpy as np
import statsmodels.api as sm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from review_loader import getFeaturesForAllReviews

def logistic_predictor_from_data(train_targets, train_regressors):
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    return predictor

def performDoc2VecClassification(trainingData, testData):
    taggedDocuments = [TaggedDocument(doc[2], [doc[1]]) for i,doc in enumerate(trainingData)]
    model = Doc2Vec(taggedDocuments, dm=1, vector_size=100, max_epochs=10)

    train_targets = [1.0 if doc[0] == 'POS' else 0.0 for doc in trainingData]

    train_regressors = [model.docvecs[doc[1]] for doc in trainingData]
    train_regressors = sm.add_constant(train_regressors)

    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_regressors = [model.infer_vector(doc[2]) for doc in testData]
    test_regressors = sm.add_constant(test_regressors)

    test_predictions = predictor.predict(test_regressors)
    print(np.rint(test_predictions))
    corrects = sum(np.rint(test_predictions) == [1.0 if doc[0] == 'POS' else 0.0 for doc in testData])
    performance = float(corrects) / len(test_predictions)

    print(performance)



features = getFeaturesForAllReviews()

trainingSplit = []
testSplit = []

trainingSplit.extend(features[0:900])
trainingSplit.extend(features[1000:1900])
testSplit.extend(features[900:1000])
testSplit.extend(features[1900:2000])

performDoc2VecClassification(features[0:1800], features[1800:2000])
