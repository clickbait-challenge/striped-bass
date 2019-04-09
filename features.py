import os
import csv
import time
import pandas as pd
import re
import nltk

'''
  ### Fields in instances.jsonl:
  { 
    "id": "<instance id>",
    "postTimestamp": "<weekday> <month> <day> <hour>:<minute>:<second> <time_offset> <year>",
    "postText": ["<text of the post with links removed>"],
    "postMedia": ["<path to a file in the media archive>"],
    "targetTitle": "<title of target article>",
    "targetDescription": "<description tag of target article>",
    "targetKeywords": "<keywords tag of target article>",
    "targetParagraphs": ["<text of the ith paragraph in the target article>"],
    "targetCaptions": ["<caption of the ith image in the target article>"]
  }
  ### Fields in truth_data.jsonl:
  {
    "id": "<instance id>",
    "truthJudgments": [<number in [0,1]>],
    "truthMean": <number in [0,1]>,
    "truthMedian": <number in [0,1]>,
    "truthMode": <number in [0,1]>,
    "truthClass": "clickbait | no-clickbait"
  }
'''

INSTANCES = 'instances.jsonl'
TRUTH = 'truth.jsonl'

FEATURES = 'features.csv'


def tokenize_words(text):
    return nltk.word_tokenize(text)


def find_pos_tags(tokens):
    return nltk.pos_tag(tokens)


def find_nnp(tups):
    return sum([1 if tup[1]=='NNP' else 0 for tup in tups])


def find_nn(tups):
    return sum([1 if tup[1]=='NN' else 0 for tup in tups])


def find_dt(tups):
    return sum([1 if tup[1]=='DT' else 0 for tup in tups])


def find_rb(tups):
    return sum([1 if tup[1]=='RB' else 0 for tup in tups])


def find_prp(tups):
    return sum([1 if tup[1]=='PRP' else 0 for tup in tups])


def find_vbd(tups):
    return sum([1 if tup[1]=='VBD' else 0 for tup in tups])


def find_vbp(tups):
    return sum([1 if tup[1]=='VBP' else 0 for tup in tups])


def good_words(tups):
    return [tup[0] for tup in tups if tup[1] in ['DT', 'RB', 'PRP']]


def char_length(line):
    return len(line)


def word_length(line):
    return len(line.split())


def distance(a, b):
    return abs(a - b)


def extractFeatures(inDir):
    print("Starting feature generation")
    start_time = time.time()

    data = pd.read_json(os.path.join(inDir, INSTANCES), dtype={'id': str}, lines=True)
    data['postText'] = data['postText'].apply(lambda x: x[0])

    data['processed_postText'] = data.postText.apply(lambda x: find_pos_tags(tokenize_words(''.join(x).lower())))
    data['processed_targetDescription'] = data.targetDescription.apply(lambda x: find_pos_tags(tokenize_words(x.lower())))
    data['processed_targetTitle'] = data.targetTitle.apply(lambda x: find_pos_tags(tokenize_words(x.lower())))

    features = pd.DataFrame()
    features['id'] = data['id']


    if (os.path.isfile(os.path.join(inDir, TRUTH))):
        print("Truth file found")
        truth_data = pd.read_json(os.path.join(inDir, TRUTH), dtype={'id': str}, lines=True)
        features = features.merge(truth_data[['id', 'truthClass']], on='id')

    fields = ['postText', 'targetTitle', 'targetDescription', 'targetKeywords']
    fields_2 = ['processed_postText', 'processed_targetDescription', 'processed_targetTitle']

    for f in fields:
        features['chars_' + f] = data[f].apply(char_length)
        features['wrds_' + f] = data[f].apply(word_length)
        features['#?_' + f] = data[f].apply(lambda x: x.count('?'))
        features['#!_' + f] = data[f].apply(lambda x: x.count('!'))
        features['##_' + f] = data[f].apply(lambda x: len(re.findall(r"(?<!#)#(?![#\s])", x)))
        features['#^\d_' + f] = data[f].apply(lambda x: len(re.findall(r"^\d.*", x)))

    for x in fields_2:
        features['count_nn' + x] = data[x].apply(find_nn)
        features['count_nnp' + x] = data[x].apply(find_nnp)
        features['count_dt' + x] = data[x].apply(find_dt)
        features['count_prp' + x] = data[x].apply(find_prp)
        features['count_rb' + x] = data[x].apply(find_rb)
        features['count_vbd' + x] = data[x].apply(find_vbd)
        features['count_vbp' + x] = data[x].apply(find_vbp)


    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            features['d_chars_' + fields[i] + '-' + fields[j]] = distance(features["chars_" + fields[i]],
                                                                          features['chars_' + fields[j]])
            features['d_wrds_' + fields[i] + '-' + fields[j]] = distance(features["wrds_" + fields[i]],
                                                                         features['wrds_' + fields[j]])

    features.to_csv(FEATURES, index=False)
    print("Feature generation took {}".format(time.time() - start_time))


if __name__ == "__main__":
    extractFeatures('data-medium')