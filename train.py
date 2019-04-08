from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
import pandas as pd

RANDOM_FORREST_CLASSIFIER = 'randomforrest.joblib'
XGBOOST_MODEL = 'xgboost.model'

def trainRandomForrest(train_file):

    # train = pd.read_csv(train_file, index_col=0)
    # train = train.sort_index()

    # truth = pd.read_json(truth_file, lines=True)
    # truth.set_index('id', inplace=True)
    # truth = truth.sort_index()
    # truth = truth[['truthClass']]
    # truth['truth'] = pd.factorize(truth['truthClass'])[0]
    # print(truth.head(), end='\n\n\n')
    # print(train.head())

    # # Drop train data columns
    # # train =  train.drop(train.columns[0], axis = 1)
    # print(len(train), len(truth))

    train = pd.read_csv(train_file)
    print(train.head())

    truth = pd.factorize(train['truthClass'])[0]
    train.drop(columns=['id','truthClass'], axis = 1, inplace = True)
    
    print(train.head())

    clf = RandomForestClassifier()
    clf.fit(train, truth)
    joblib.dump(clf, RANDOM_FORREST_CLASSIFIER)
    return clf


def trainXGBoost(train_file):
    train = pd.read_csv(train_file)
    print(train.head())
    truth = pd.factorize(train['truthClass'])[0]
    train.drop(columns=['id','truthClass'], axis = 1, inplace = True)
    print(train.head())

    params = {
        'objective': 'reg:logistic'
    }

    dtrain = xgb.DMatrix(train, label = truth)
    model = xgb.train(params, dtrain)
    
    model.save_model(XGBOOST_MODEL)

    return model

