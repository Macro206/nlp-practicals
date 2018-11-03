import os
import porter_stemmer

p = porter_stemmer.PorterStemmer()

positiveFileList = sorted(list(filter(lambda s: s.endswith(".tag"), os.listdir("./POS"))))
negativeFileList = sorted(list(filter(lambda s: s.endswith(".tag"), os.listdir("./NEG"))))

for f in positiveFileList:
    infile = open("./POS/" + f, 'r')

    outputString = ''

    while 1:
        output = ''
        word = ''
        line = infile.readline()
        if line == '':
            break
        for c in line:
            if c.isalpha():
                word += c.lower()
            else:
                if word:
                    output += p.stem(word, 0,len(word)-1)
                    word = ''
                output += c.lower()

        outputString += output

    outfile = open("./POS_STEM/" + f, 'w')
    outfile.write(outputString)

    infile.close()

for f in negativeFileList:
    infile = open("./NEG/" + f, 'r')

    outputString = ''

    while 1:
        output = ''
        word = ''
        line = infile.readline()
        if line == '':
            break
        for c in line:
            if c.isalpha():
                word += c.lower()
            else:
                if word:
                    output += p.stem(word, 0,len(word)-1)
                    word = ''
                output += c.lower()

        outputString += output

    outfile = open("./NEG_STEM/" + f, 'w')
    outfile.write(outputString)

    infile.close()
