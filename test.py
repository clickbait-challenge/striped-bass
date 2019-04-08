from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
import pandas as pd
from train import RANDOM_FORREST_CLASSIFIER, XGBOOST_MODEL

'''
  ### Fields your classifier should write into a results.jsonl file
  {
    "id": "<instance id>",
    "clickbaitScore": <number in [0,1]>
  }
'''

def testRandomForrest(test_file):
    test = pd.read_csv(test_file)
    ids = test['id']
    test.drop(columns=['id','truthClass'], axis = 1, inplace = True)

    clf = joblib.load(RANDOM_FORREST_CLASSIFIER)
    predictions = clf.predict(test)

    results = pd.DataFrame({'id':ids, 'clickbaitScore':predictions})
    results.to_json('results.jsonl', orient='records', lines=True)

    return results

# testRandomForrest('data-small/features.csv')

def testXGBoost(test_file):
    test = pd.read_csv(test_file)
    ids = test['id']
    test.drop(columns=['id','truthClass'], axis = 1, inplace = True)
    dtest = xgb.DMatrix(test)

    booster = xgb.Booster()
    booster.load_model(XGBOOST_MODEL)

    predictions = booster.predict(dtest)
    
    results = pd.DataFrame({'id':ids, 'clickbaitScore':predictions})
    results.to_json('results.jsonl', orient='records', lines=True)

    return results

# testXGBoost('data-small/features.csv')

