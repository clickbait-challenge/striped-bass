from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb

RANDOM_FORREST_CLASSIFIER = 'randomforrest.joblib'
XGBOOST_MODEL = 'xgboost.model'

def trainRandomForrest(train, truths):
    clf = RandomForestClassifier()
    clf.fit(train, truths)
    joblib.dump(clf, RANDOM_FORREST_CLASSIFIER)
    return clf

def evaluateRandomForrest(test):
    clf = joblib.load(RANDOM_FORREST_CLASSIFIER)
    predictions = clf.predict(test)
    return predictions


def trainXGBoost(train):
    params = {
    }
    dtrain = xgb.DMatrix(train)
    model = xgb.train(params, dtrain)
    
    model.save_model(XGBOOST_MODEL)

    return model

def evaluateXGBoost(test):
    booster = xgb.Booster()
    booster.load_model(XGBOOST_MODEL)

    dtest = xgb.DMatrix(test)
    
    predictions = booster.predict(dtest)

    return predictions
