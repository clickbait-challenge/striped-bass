from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
import pandas as pd

RANDOM_FORREST_CLASSIFIER = 'randomforrest.joblib'
XGBOOST_MODEL = 'xgboost.model'

def trainRandomForrest(train_file):
    train = pd.read_csv(train_file)
    truth = pd.factorize(train['truthClass'])[0]
    train.drop(columns=['id','truthClass'], axis = 1, inplace = True)

    clf = RandomForestClassifier()
    clf.fit(train, truth)
    joblib.dump(clf, RANDOM_FORREST_CLASSIFIER)
    return clf


def trainXGBoost(train_file):
    train = pd.read_csv(train_file)
    truth = pd.factorize(train['truthClass'])[0]
    train.drop(columns=['id','truthClass'], axis = 1, inplace = True)

    params = {
        'objective': 'reg:logistic'
    }

    dtrain = xgb.DMatrix(train, label = truth)
    model = xgb.train(params, dtrain)
    
    model.save_model(XGBOOST_MODEL)

    return model

