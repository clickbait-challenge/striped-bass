from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
import pandas as pd
import time

RANDOM_FORREST_CLASSIFIER = 'randomforrest.joblib'
XGBOOST_MODEL = 'xgboost.model'


def trainRandomForrest(train_file, outDir):
    print("Starting Training RandomForrest")
    start_time = time.time()

    train = pd.read_csv(train_file)
    truth = pd.factorize(train['truthClass'])[0]
    train.drop(columns=['id','truthClass'], axis = 1, inplace = True)

    clf = RandomForestClassifier()
    clf.fit(train, truth)
    joblib.dump(clf, RANDOM_FORREST_CLASSIFIER)

    print("Random Forrest training {}".format(time.time() - start_time))
    return clf


def trainXGBoost(train_file):
    print("Starting Training XGBoost")
    start_time = time.time()

    train = pd.read_csv(train_file)
    truth = pd.factorize(train['truthClass'])[0]
    train.drop(columns=['id','truthClass'], axis = 1, inplace = True)

    params = {
        'objective': 'reg:logistic'
    }

    dtrain = xgb.DMatrix(train, label = truth)
    model = xgb.train(params, dtrain)
    
    model.save_model(XGBOOST_MODEL)
    
    print("XGBoost training took {}".format(time.time() - start_time))
    return model

