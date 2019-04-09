import os
import json_lines
import csv
import time
import pandas as pd
import re
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

global truth_data
truth_data = None

def compute_features(instance, truth_data):
    features = [instance['id']]
    
    fields = ['postText', 'targetTitle', 'targetDescription', 'targetKeywords']
    for field in fields:
      field_data = ''
      # postText is an array with one field, so extract data
      if isinstance(instance[field], list):
        field_data = instance[field][0]
      else:
        field_data = instance[field]
      
      # Single field features
      features.append(char_length(field_data))
      features.append(word_length(field_data))
      features.append(field_data.count('?'))
      features.append(field_data.count('!'))
      # Amount of hashtags
      features.append(len(re.findall(r"(?<!#)#(?![#\s])", field_data)))
      # Starts with number or not
      features.append(len(re.findall(r"^\d.*", field_data)))

    # Two field features

    #Word / char distance
    for i in range(len(fields)):
        for j in range(i+1, len(fields)):
            a = fields[i]
            b = fields[j]

            if isinstance(instance[a], list):
                features.append(distance(char_length(instance[a][0]), char_length(instance[b])))
                features.append(distance(word_length(instance[a][0]), word_length(instance[b])))

            else :
                features.append(distance(char_length(instance[a]), char_length(instance[b])))
                features.append(distance(word_length(instance[a]), word_length(instance[b])))
    
    

    # Add truth if available
    if truth_data is not None:
      features.append(truth_data.loc[truth_data['id'] == instance['id'], 'truthClass'].item())

    return features

def computeFeatureHeader():
  header = ['id']
  fields = ['postText', 'targetTitle', 'targetDescription', 'targetKeywords']

  for f in fields:
    header.append("chars_" + f)
    header.append("words_" + f)
    # Amount of ?
    header.append("#?_" + f)
    # Amount of !
    header.append("#!_" + f)
    # Amount of hashtags
    header.append("#hashtags_" + f)
    # Starts with number or not
    header.append("#^\\d_" + f)

  for i in range(len(fields)):
      for j in range(i+1, len(fields)):
        header.append("diff_chars_"+fields[i]+'-'+fields[j])
        header.append("diff_words_"+fields[i]+'-'+fields[j])

  header.append("truthClass")

  return header



def char_length(line):
  return len(line)

def word_length(line):
  return len(line.split())

def distance(a, b):
  return abs(a - b)

def extractFeatures(inDir):
  print("Starting feature generation")

  if (os.path.isfile(os.path.join(inDir, TRUTH))):
    print("Truth file found")
    truth_data = pd.read_json(os.path.join(inDir, TRUTH), dtype={'id':str}, lines=True)

  currentLine = 1
  start_time = time.time()

  with open(os.path.join(inDir, INSTANCES), 'rb') as file :
      with open(FEATURES, "w") as out:
          out_writer = csv.writer(out)
          out_writer.writerow(computeFeatureHeader())
          for line in json_lines.reader(file):
              if not line:
                  break

              features = compute_features(line, truth_data)
              out_writer.writerow(features)

              currentLine += 1
              # if (currentLine % 1000) == 0:
              print("Processed {} lines".format(currentLine), end="\r")

  print("Feature generation took {}".format(time.time() - start_time))



if __name__ == "__main__":
    extractFeatures('data-small')
    
    





