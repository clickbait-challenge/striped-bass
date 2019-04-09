from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


PATH_TO_INPUT = 'instances.jsonl'
TRUTH = 'truth.jsonl'
PATH_TO_FILE = 'features.csv'

analyser = SentimentIntensityAnalyzer()


#function that assign 1 to positive text 
def sentiment_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #return list(snt.keys())[list(snt.values()).index(max(snt.values()))]
    return 1 if score['pos'] > score['neg'] else 0

def main():
    df = pd.read_json(PATH_TO_INPUT, dtype={'id': str}, lines=True)
    df['postText'] = df['postText'].apply(lambda x: x[0])
    features = pd.DataFrame()

    features['id'] = df['id']

    fields = ['postText', 'targetTitle', 'targetDescription']
   
    for f in fields[0:2]:
        features['sentiment' + f] = df[f].apply(sentiment_scores)

    print(features)




if __name__ == '__main__':
    main()
