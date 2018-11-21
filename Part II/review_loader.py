import os

options = {
    "shouldUseStemmedReviews": False,
    "shouldUseBigrams": False,
    "shouldUseUnigramsAndBigrams": False,
    "shouldUsePresence": False,
    "shouldUseCutoffs": False,
}

amazon_datasets = {
    "digital_music": "Digital_Music_5",
    "gardening": "Patio_Lawn_and_Garden_5",
    "video_games": "Video_Games_5",
    "food": "Grocery_and_Gourmet_Food_5",
    "musical_instruments": "Musical_Instruments_5",
    "office": "Office_Products_5",
    "video": "Amazon_Instant_Video_5"
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

def getFeaturesForAllAmazonReviews(dataset):
    positiveDirPath = '/Users/Matteo/Desktop/amazon_test_data/' + amazon_datasets[dataset] + '/POS'
    negativeDirPath = '/Users/Matteo/Desktop/amazon_test_data/' + amazon_datasets[dataset] + '/NEG'

    positiveFileList = sorted(list(filter(lambda s: s.endswith(".conll"), os.listdir(positiveDirPath))))
    negativeFileList = sorted(list(filter(lambda s: s.endswith(".conll"), os.listdir(negativeDirPath))))

    features = []

    for f in positiveFileList:
        features.append(("POS", 'pos_' + f, getFeaturesFromFile(positiveDirPath + "/" + f)))

    for f in negativeFileList:
        features.append(("NEG", 'neg_' + f, getFeaturesFromFile(negativeDirPath + "/" + f)))

    return features
