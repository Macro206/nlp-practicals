import os

shouldUseStemmedReviews = False
shouldUseBigrams = False

### FUNCTIONS TO LOAD FILES INTO REQUIRED FORMAT ###

def getFeaturesFromFile(filePath):
    with open(filePath) as f:
        fileString = f.read()
        features = fileString.split("\n")

    filteredFeatures = list(filter(lambda s: s != '', features))  # TODO: Mention the removal of empty strings

    if shouldUseBigrams:
        bigramFeatures = []

        for i in range(1,len(filteredFeatures)):
            bigramFeatures.append(filteredFeatures[i-1] + ":" + filteredFeatures[i])

        return bigramFeatures


    return filteredFeatures

def getFeaturesForAllReviews():
    positiveDirPath = "./POS_STEM" if shouldUseStemmedReviews else "./POS"
    negativeDirPath = "./NEG_STEM" if shouldUseStemmedReviews else "./NEG"

    if shouldUseStemmedReviews:
        print "Using stemmed reviews..."
        print ""

    if shouldUseBigrams:
        print "Using bigrams..."
        print ""

    positiveFileList = sorted(list(filter(lambda s: s.endswith(".tag"), os.listdir(positiveDirPath))))
    negativeFileList = sorted(list(filter(lambda s: s.endswith(".tag"), os.listdir(negativeDirPath))))

    features = []

    for f in positiveFileList:
        features.append(("POS", f, getFeaturesFromFile(positiveDirPath + "/" + f)))

    for f in negativeFileList:
        features.append(("NEG", f, getFeaturesFromFile(negativeDirPath + "/" + f)))

    return features
