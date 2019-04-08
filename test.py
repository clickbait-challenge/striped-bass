from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
import pandas as pd

def evaluateRandomForrest(test_file):
    # test = pd.read_json(test_file, lines=True)


    clf = joblib.load(RANDOM_FORREST_CLASSIFIER)
    predictions = clf.predict(test)

    # write to file

    return predictions

def evaluateXGBoost(test_file):
    test = pd.read_csv(test_file)
    dtest = xgb.DMatrix(test)

    booster = xgb.XGBClassifier()
    booster.load_model(XGBOOST_MODEL)

    predictions = booster.predict(dtest)

    return predictions

