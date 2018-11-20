import os

options = {
    "shouldUseStemmedReviews": False,
    "shouldUseBigrams": False,
    "shouldUseUnigramsAndBigrams": False,
    "shouldUsePresence": False,
    "shouldUseCutoffs": False,
}

### FUNCTIONS TO LOAD FILES INTO REQUIRED FORMAT ###

def getFeaturesFromFile(filePath):
    with open(filePath) as f:
        fileString = f.read()
        features = fileString.split("\n")

    filteredFeatures = list(filter(lambda s: s != '', features))  # TODO: Mention the removal of empty strings

    if options["shouldUseBigrams"]:
        bigramFeatures = []

        for i in range(1,len(filteredFeatures)):
            bigramFeatures.append(filteredFeatures[i-1] + ":" + filteredFeatures[i])

        if options["shouldUseUnigramsAndBigrams"]:
            filteredFeatures.extend(bigramFeatures)
            return filteredFeatures
        else:
            return bigramFeatures


    return filteredFeatures

def getFeaturesForAllReviews():
    positiveDirPath = "./POS_STEM" if options["shouldUseStemmedReviews"] else "./POS"
    negativeDirPath = "./NEG_STEM" if options["shouldUseStemmedReviews"] else "./NEG"

    if options["shouldUseStemmedReviews"]:
        print("Using stemmed reviews...")

    if options["shouldUseBigrams"]:
        print("Using bigrams...")

    if options["shouldUsePresence"]:
        print("Using presence...")

    if options["shouldUseCutoffs"]:
        print("Using cutoffs...")

    print("")

    positiveFileList = sorted(list(filter(lambda s: s.endswith(".tag"), os.listdir(positiveDirPath))))
    negativeFileList = sorted(list(filter(lambda s: s.endswith(".tag"), os.listdir(negativeDirPath))))

    features = []

    for f in positiveFileList:
        features.append(("POS", f, getFeaturesFromFile(positiveDirPath + "/" + f)))

    for f in negativeFileList:
        features.append(("NEG", f, getFeaturesFromFile(negativeDirPath + "/" + f)))

    return features

def getFeaturesForAllAmazonReviews():
    positiveDirPath = '/Users/Matteo/Desktop/amazon_test_data/POS'
    negativeDirPath = '/Users/Matteo/Desktop/amazon_test_data/NEG'

    positiveFileList = sorted(list(filter(lambda s: s.endswith(".conll"), os.listdir(positiveDirPath))))
    negativeFileList = sorted(list(filter(lambda s: s.endswith(".conll"), os.listdir(negativeDirPath))))

    features = []

    for f in positiveFileList:
        features.append(("POS", 'pos_' + f, getFeaturesFromFile(positiveDirPath + "/" + f)))

    for f in negativeFileList:
        features.append(("NEG", 'neg_' + f, getFeaturesFromFile(negativeDirPath + "/" + f)))

    return features
