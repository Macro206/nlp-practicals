import json
import numpy as np

positiveReviews = []
negativeReviews = []

with open('/Users/Matteo/Desktop/amazon_test_data/Digital_Music_5.json') as f:
    jsonStrings = f.read().split('\n')

    for s in jsonStrings:
        if s == '':
            continue

        obj = json.loads(s)
        if float(obj['overall']) > 3.1:
            positiveReviews.append(obj['reviewText'])
        elif float(obj['overall']) < 2.9:
            negativeReviews.append(obj['reviewText'])


finalPositiveReviews = np.random.choice(positiveReviews, 100, replace=False)
finalNegativeReviews = np.random.choice(negativeReviews, 100, replace=False)

i = 0
for r in finalPositiveReviews:
    with open('/Users/Matteo/Desktop/amazon_test_data/POS_UNTOK/' + str(i) + '.txt', 'w') as f:
        f.write(r)

    i += 1

i = 0
for r in finalNegativeReviews:
    with open('/Users/Matteo/Desktop/amazon_test_data/NEG_UNTOK/' + str(i) + '.txt', 'w') as f:
        f.write(r)

    i += 1


with open('/Users/Matteo/Desktop/amazon_test_data/positive_file_list.txt', 'w') as f:
    for i in range(0,100):
        f.write('/Users/Matteo/Desktop/amazon_test_data/POS_UNTOK/' + str(i) + '.txt' + '\n')

with open('/Users/Matteo/Desktop/amazon_test_data/negative_file_list.txt', 'w') as f:
    for i in range(0,100):
        f.write('/Users/Matteo/Desktop/amazon_test_data/NEG_UNTOK/' + str(i) + '.txt' + '\n')
