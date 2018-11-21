import json
import numpy as np
import subprocess
import os
import shutil

dataset_name = "Amazon_Instant_Video_5"

core_nlp_path = '/Users/Matteo/Downloads/stanford-corenlp-full-2018-10-05'

amazon_data_root = '/Users/Matteo/Desktop/amazon_test_data/'
test_data_path = amazon_data_root + dataset_name + '/'

nReviews = 100

positiveReviews = []
negativeReviews = []

foldersToCreate = ['POS', 'POS_UNTOK', 'NEG', 'NEG_UNTOK']

for f in foldersToCreate:
    folderPath = test_data_path + f

    if os.path.exists(folderPath):
        shutil.rmtree(folderPath)
    os.mkdir(folderPath)

with open(test_data_path + dataset_name + '.json') as f:
    jsonStrings = f.read().split('\n')

    for s in jsonStrings:
        if s == '':
            continue

        obj = json.loads(s)
        if float(obj['overall']) > 3.1:
            positiveReviews.append(obj['reviewText'])
        elif float(obj['overall']) < 2.9:
            negativeReviews.append(obj['reviewText'])


finalPositiveReviews = np.random.choice(positiveReviews, nReviews, replace=False)
finalNegativeReviews = np.random.choice(negativeReviews, nReviews, replace=False)

i = 0
for r in finalPositiveReviews:
    with open(test_data_path + 'POS_UNTOK/' + str(i) + '.txt', 'w') as f:
        f.write(r)

    i += 1

i = 0
for r in finalNegativeReviews:
    with open(test_data_path + 'NEG_UNTOK/' + str(i) + '.txt', 'w') as f:
        f.write(r)

    i += 1


with open(test_data_path + 'positive_file_list.txt', 'w') as f:
    for i in range(0,nReviews):
        f.write(test_data_path + 'POS_UNTOK/' + str(i) + '.txt' + '\n')

with open(test_data_path + 'negative_file_list.txt', 'w') as f:
    for i in range(0,nReviews):
        f.write(test_data_path + 'NEG_UNTOK/' + str(i) + '.txt' + '\n')

p1 = subprocess.Popen(['java', '-cp', 'stanford-corenlp-3.9.2.jar', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                       '-annotators', 'tokenize,ssplit', '-filelist', test_data_path+'positive_file_list.txt',
                       '-outputFormat', 'conll', '-output.columns', 'word', '-outputDirectory',
                       test_data_path+'POS'], cwd=core_nlp_path)
p1.wait()

p2 = subprocess.Popen(['java', '-cp', 'stanford-corenlp-3.9.2.jar', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                       '-annotators', 'tokenize,ssplit', '-filelist', test_data_path+'negative_file_list.txt',
                       '-outputFormat', 'conll', '-output.columns', 'word', '-outputDirectory',
                       test_data_path+'NEG'], cwd=core_nlp_path)
p2.wait()
