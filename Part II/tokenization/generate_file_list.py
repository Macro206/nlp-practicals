import os
from shutil import copyfile

def moveReviewsForFolder(folderPath,i):
    files = os.listdir(folderPath)

    for f in files:
        copyfile(folderPath+f, '/Users/Matteo/Desktop/doc2vec_training_data_untokenized/'+str(i)+'__'+f)


def moveReviews():
    corpusRoot = "/Users/Matteo/Desktop/aclImdb/"

    paths = ["train/neg/",
             "train/pos/",
             "train/unsup/",
             "test/neg/",
             "test/pos/"]

    for i in range(0,len(paths)):
        p = paths[i]
        moveReviewsForFolder(corpusRoot+p,i)


def generateFileList():
    files = os.listdir('/Users/Matteo/Desktop/doc2vec_training_data_untokenized/')

    with open('./filelist.txt', 'w') as fileList:
        for f in files:
            fileList.write('/Users/Matteo/Desktop/doc2vec_training_data_untokenized/'+f+'\n')


generateFileList()
