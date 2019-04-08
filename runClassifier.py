import sys
import getopt
import os

from features import extractFeatures
from train import trainRandomForrest, trainXGBoost, RANDOM_FORREST_CLASSIFIER, XGBOOST_MODEL
from test import testRandomForrest, testXGBoost


def main():

    argv = sys.argv[1:]
    # print(argv)

    try:
        opts, args = getopt.getopt(argv, 'i:o:c:', [])
        print(opts)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inDir = ''
    outDir = ''
    classifier = ''


    for opt, arg in opts:
        if opt == '-i':
            if not os.path.exists(arg):
                print('Aborting, can not find infile:', arg)
                sys.exit(2)
            inDir = arg
        elif opt == '-o':
            outDir = arg
        else:
            if arg not in ('xgboost', 'randomforrest'):
                print('Aborting, invalid classifier must be "xgboost" or "randomforrest"')
                sys.exit(2)
            classifier = arg

    
    extractFeatures(inDir)
    
    if classifier == 'randomforrest':
        if not os.path.exists(RANDOM_FORREST_CLASSIFIER):
            print('Could not find trained RandomForrestClassifier')
            sys.exit(2)
        testRandomForrest('features.csv', outDir)

    if classifier == 'xgboost':
        if not os.path.exists(XGBOOST_MODEL):
            print('Can not find trained XGBoost model')
            sys.exit(2)
        testXGBoost('features.csv', outDir)
    
    

if __name__ == "__main__":
    main()