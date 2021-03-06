from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
import pandas as pd
import time
import os

from train import RANDOM_FORREST_CLASSIFIER, XGBOOST_MODEL

'''
  ### Fields your classifier should write into a results.jsonl file
  {
    "id": "<instance id>",
    "clickbaitScore": <number in [0,1]>
  }
'''

def testRandomForrest(test_file, outDir="./", feat_selection = []):
    print("Starting RandomForrest")
    start_time = time.time()

    test = pd.read_csv(test_file)
    
    ids = test['id']
    test.drop(columns=['id'], axis = 1, inplace = True)

    if 'truthClass' in test.columns:
      test.drop(columns=['truthClass'], axis = 1, inplace = True)


    if len(feat_selection) is not 0:
      test = test[feat_selection]
        
    clf = joblib.load(RANDOM_FORREST_CLASSIFIER)
    predictions = clf.predict(test)

    results = pd.DataFrame({'id':ids, 'clickbaitScore':predictions})
    results['id'] = results['id'].astype(str)
    results.to_json(os.path.join(outDir, 'results.jsonl'), orient='records', lines=True)

    print("Random Forrest training {}".format(time.time() - start_time))
    return results

def testXGBoost(test_file, outDir="./", feat_selection = []):
    print("Starting XGBoost")
    start_time = time.time()

    test = pd.read_csv(test_file)
    ids = test['id']
    test.drop(columns=['id'], axis = 1, inplace = True)

    if 'truthClass' in test.columns:
      test.drop(columns=['truthClass'], axis = 1, inplace = True)
  
    if len(feat_selection) is not 0:
        test = test[feat_selection]

    dtest = xgb.DMatrix(test)

    booster = xgb.Booster()
    booster.load_model(XGBOOST_MODEL)

    predictions = booster.predict(dtest)
    
    results = pd.DataFrame({'id':ids, 'clickbaitScore':predictions})
    results['id'] = results['id'].astype(str)
    results.to_json(os.path.join(outDir, 'results.jsonl'), orient='records', lines=True)

    print("XGBoost took {}".format(time.time() - start_time))
    return results

