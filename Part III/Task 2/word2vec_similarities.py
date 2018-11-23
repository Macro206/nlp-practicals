import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('/Users/Matteo/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

features = []

with open('./text2.txt.conll', 'r') as f:
    lines = f.read().split('\n')

    for l in lines:
        if l == '':
            continue
        features.append(l)

tagggedFeatures = {}

for word in features:
    try:
        similarity = model.n_similarity(['pay'], [word])
    except:
        continue

    tagggedFeatures[word] = similarity



print("Most similar words:")

kvPairs = []

for key in tagggedFeatures:
    kvPairs.append((key, tagggedFeatures[key]))

kvPairs.sort(key=lambda x: x[1], reverse=True)

for i in range(0,11):
    print(str(kvPairs[i][0]) + ': ' + str(kvPairs[i][1]))

print('supplementary' + ': ' + str(tagggedFeatures['supplementary']))
