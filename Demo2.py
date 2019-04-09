from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


PATH_TO_INPUT = 'instances.jsonl'
TRUTH = 'truth.jsonl'
PATH_TO_FILE = 'features.csv'

analyser = SentimentIntensityAnalyzer()

def sentiment_scores(sentence):
    snt = analyser.polarity_scores(sentence)
    return list(snt.keys())[list(snt.values()).index(max(snt.values()))]
    #return 1 if score['pos'] > score['neg'] else 0

def main():
    df = pd.read_json(PATH_TO_INPUT, dtype={'id': str}, lines=True)
    df['postText'] = df['postText'].apply(lambda x: x[0])
    features = pd.DataFrame()

    features['id'] = df['id']

    fields = ['postText', 'targetTitle', 'targetDescription']
    data3 = pd.DataFrame()

    data3['id'] = df['id']

    for f in fields:
        data3['sentiment' +f] = df[f].apply(sentiment_scores)


    #df['sentiment'] = df.postText.apply(sentiment_scores)
    print(print(data3.head()))




if __name__ == '__main__':
    main()
