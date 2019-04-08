from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
import pandas as pd
import time
import sys

from features import extractFeatures, FEATURES

RANDOM_FORREST_CLASSIFIER = 'randomforrest.joblib'
XGBOOST_MODEL = 'xgboost.model'


def trainRandomForrest(train_file):
    print("Starting Training RandomForrest")
    start_time = time.time()

    train = pd.read_csv(train_file)
    train.replace({"truthClass": {"no-clickbait":0, "clickbait":1}}, inplace = True)
    truth = train['truthClass']
    train.drop(columns=['id','truthClass'], axis = 1, inplace = True)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train, truth)
    joblib.dump(clf, RANDOM_FORREST_CLASSIFIER)

    print("Random Forrest training {}".format(time.time() - start_time))
    return clf


def trainXGBoost(train_file):
    print("Starting Training XGBoost")
    start_time = time.time()

    train = pd.read_csv(train_file)
    train.replace({"truthClass": {"no-clickbait":0, "clickbait":1}}, inplace = True)
    truth = train['truthClass']

    train.drop(columns=['id','truthClass'], axis = 1, inplace = True)

    params = {
        'objective': 'reg:logistic'
    }

    dtrain = xgb.DMatrix(train, label = truth)
    model = xgb.train(params, dtrain)
    
    model.save_model(XGBOOST_MODEL)
    
    print("XGBoost training took {}".format(time.time() - start_time))
    return model

def trainClassifiers():
    argv = sys.argv[1:]

    extractFeatures(argv[0])

    trainXGBoost(FEATURES)
    trainRandomForrest(FEATURES)


if __name__ == "__main__":
    trainClassifiers()