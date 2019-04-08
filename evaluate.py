from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

import xgboost as xgb
import pandas as pd
import time
import os
import sys

from train import RANDOM_FORREST_CLASSIFIER, XGBOOST_MODEL
from features import extractFeatures, FEATURES
from train import trainRandomForrest, trainXGBoost
from test import testRandomForrest, testXGBoost

EVAL_TRAIN = 'eval-train.csv'
EVAL_TEST = 'eval-test.csv'

def divide_zero_is_999(n, d):
  return n/d  if d else 999

def evaluateResults(predicted, actual, test):
  truePos = 0
  falsePos = 0
  trueNeg = 0
  falseNeg = 0
  for i, v in pd.Series(predicted).items():
    if v == 0 and v == actual[i]:
      trueNeg += 1
    if v == 1 and v == actual[i]:
      truePos += 1
    if v == 0 and v != actual[i]:
      falseNeg += 1
    if v == 1 and v != actual[i]:
      falsePos += 1
  
  # print('accuracy', (pd.Series(predicted) == actual).sum()/len(actual))

  print("accuracy", (truePos+trueNeg)/len(actual))
  precision = divide_zero_is_999(truePos, truePos+falsePos)
  print("precision", precision)
  recall =  divide_zero_is_999(truePos, truePos+falseNeg)
  print("recall", recall)
  print("f1", divide_zero_is_999(2 * (precision*recall) , precision + recall ))

  print(pd.crosstab(test['truthClass'], predicted, rownames=['actual'], colnames=['predicted']))



def evaluateRandomForrest(test, train):
    print("Starting RandomForrest eval")
    start_time = time.time()

    clf = trainRandomForrest(EVAL_TRAIN)
    results = testRandomForrest(EVAL_TEST)

    predicted = results['clickbaitScore']
    actual = pd.factorize(test['truthClass'])[0]

    evaluateResults(predicted, actual, test)
    # Feature importance
    print(list(zip(train.drop(['id', 'truthClass'], axis=1), clf.feature_importances_)))

    print("Random Forrest eval took {}".format(time.time() - start_time))


def evaluateXGBoost(test, train):
    print("Starting XGBoost eval")
    start_time = time.time()

    clf = trainXGBoost(EVAL_TRAIN)
    results = testXGBoost(EVAL_TEST)

    predicted = results['clickbaitScore']
    predictedRounded = predicted.round()

    actual = pd.factorize(test['truthClass'])[0]

    evaluateResults(predictedRounded, actual, test)
  
    print("XGBoost eval took {}".format(time.time() - start_time))

def evaluateClassifiers():
    argv = sys.argv[1:]

    extractFeatures(argv[0])

    data = pd.read_csv(FEATURES)
    train, test = train_test_split(data, test_size = 0.2, random_state=1)

    train.to_csv(EVAL_TRAIN)
    test.to_csv(EVAL_TEST)

    evaluateRandomForrest(test, train)
    evaluateXGBoost(test, train)


if __name__ == "__main__":
    evaluateClassifiers()