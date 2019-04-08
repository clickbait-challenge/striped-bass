import os
import json_lines
import csv
import time
import pandas as pd
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

  ### Fields in truth.jsonl:
  {
    "id": "<instance id>",
    "truthJudgments": [<number in [0,1]>],
    "truthMean": <number in [0,1]>,
    "truthMedian": <number in [0,1]>,
    "truthMode": <number in [0,1]>,
    "truthClass": "clickbait | no-clickbait"
  }

  ### Fields your classifier should write into a results.jsonl file
  {
    "id": "<instance id>",
    "clickbaitScore": <number in [0,1]>
  }
'''

FOLDER = 'data-small'
INSTANCES = 'instances.jsonl'
TRUTH = 'truth.jsonl'

FEATURES = 'features.csv'

truth = None
if (os.path.isfile(os.path.join(FOLDER, TRUTH))):
  truth = pd.read_json(os.path.join(FOLDER, TRUTH), dtype={'id':str}, lines=True)

def compute_features(instance):
    features = [instance['id']]

    #Char lengths
    features.append(char_length(instance['postText'][0]))
    features.append(char_length(instance['targetTitle']))
    features.append(char_length(instance['targetDescription']))
    features.append(char_length(instance['targetKeywords']))

    #Word lenghts
    features.append(word_length(instance['postText'][0]))
    features.append(word_length(instance['targetTitle']))
    features.append(word_length(instance['targetDescription']))
    features.append(word_length(instance['targetKeywords']))

    #Word / char distance
    # fields = ['postText', 'targetTitle', 'targetDescription', 'targetKeywords']
    # for i in range(len(fields)):
    #     for j in range(i+1, len(fields)):
    #         a = fields[i]
    #         b = fields[j]

    #         if isinstance(instance[a], list):
    #             features.append(distance(char_length(instance[a][0]), char_length(instance[b])))
    #             features.append(distance(word_length(instance[a][0]), word_length(instance[b])))

    #         else :
    #             features.append(distance(char_length(instance[a]), char_length(instance[b])))
    #             features.append(distance(word_length(instance[a]), word_length(instance[b])))
    
    if truth is not None:
      features.append(truth.loc[truth['id'] == instance['id'], 'truthClass'].item())

    return features

def char_length(line):
    return len(line)

def word_length(line):
    return len(line.split())

def distance(a, b):
    return abs(a - b)

print(os.path.join(FOLDER,INSTANCES, os.path.join(FOLDER,FEATURES)))

currentLine = 1

start_time = time.time()

with open(os.path.join(FOLDER,INSTANCES), 'rb') as file :
    with open(os.path.join(FOLDER,FEATURES), "w") as out:
        out_writer = csv.writer(out)
        out_writer.writerow(['id', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'truthClass'])
        for line in json_lines.reader(file):
            if not line:
                break

            features = compute_features(line)
            out_writer.writerow(features)

            currentLine += 1
            if (currentLine % 100) == 0:
               print("Processed {} lines".format(currentLine), end="\r")


print("Took {}".format(time.time() - start_time))

    
    





