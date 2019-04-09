import sys
import nltk
import numpy as np
import pandas as pd

PATH_TO_INPUT = 'instances.jsonl'
TRUTH = 'truth.jsonl'
PATH_TO_FILE = 'features.csv'


def tokenize_words(text):
    return nltk.word_tokenize(text)


def find_pos_tags(tokens):
    return nltk.pos_tag(tokens)


def find_nnp(tups):
    return sum([1 if tup[1]=='NNP' else 0 for tup in tups])

def find_nn(tups):
    return sum([1 if tup[1]=='NN' else 0 for tup in tups])

def find_dt(tups):
    return sum([1 if tup[1]=='DT' else 0 for tup in tups])

def find_rb(tups):
    return sum([1 if tup[1]=='RB' else 0 for tup in tups])

def find_prp(tups):
    return sum([1 if tup[1]=='PRP' else 0 for tup in tups])

def find_vbd(tups):
    return sum([1 if tup[1]=='VBD' else 0 for tup in tups])

def find_vbp(tups):
    return sum([1 if tup[1]=='VBP' else 0 for tup in tups])


def good_words(tups):
    return [tup[0] for tup in tups if tup[1] in ['DT', 'RB', 'PRP']]


def main():
    df = pd.read_json(PATH_TO_INPUT, dtype={'id':str}, lines=True)

    # Text preprocessing and part-of-speech tagging.
    df['processed_postText'] = df.postText.apply(lambda x: find_pos_tags(tokenize_words(''.join(x).lower())))
    df['processed_targetDescription'] = df.targetDescription.apply(lambda x: find_pos_tags(tokenize_words(x.lower())))
    df['processed_targetTitle'] = df.targetTitle.apply(lambda x: find_pos_tags(tokenize_words(x.lower())))

    # Count NNP.
    df['nnp_count_postText'] = df.processed_postText.apply(find_nnp)
    df['nnp_count_targetDescription'] = df.processed_targetDescription.apply(find_nnp)
    df['nnp_count_targetTitle'] = df.processed_targetTitle.apply(find_nnp)

    # Count NN,
    df['nn_count_postText'] = df.processed_postText.apply(find_nn)
    df['nn_count_targetDescription'] = df.processed_targetDescription.apply(find_nn)
    df['nn_count_targetTitle'] = df.processed_targetTitle.apply(find_nn)

    # Count DT.
    df['dt_count_postText'] = df.processed_postText.apply(find_dt)
    df['dt_count_targetDescription'] = df.processed_targetDescription.apply(find_dt)
    df['dt_count_targetTitle'] = df.processed_targetTitle.apply(find_dt)

    #Count RB
    df['rb_count_postText'] = df.processed_postText.apply(find_rb)
    df['rb_count_targetDescription'] = df.processed_targetDescription.apply(find_rb)
    df['rb_count_targetTitle'] = df.processed_targetTitle.apply(find_rb)

    #Count_PRP
    df['prp_count_postText'] = df.processed_postText.apply(find_prp)
    df['prp_count_targetDescription'] = df.processed_targetDescription.apply(find_prp)
    df['prp_count_targetTitle'] = df.processed_targetTitle.apply(find_prp)

    # Count_VBD
    df['vbd_count_postText'] = df.processed_postText.apply(find_vbd)
    df['vbd_count_targetDescription'] = df.processed_targetDescription.apply(find_vbd)
    df['vbd_count_targetTitle'] = df.processed_targetTitle.apply(find_vbd)

    # Count_VBP
    df['vbp_count_postText'] = df.processed_postText.apply(find_vbp)
    df['vbp_count_targetDescription'] = df.processed_targetDescription.apply(find_vbp)
    df['vbp_count_targetTitle'] = df.processed_targetTitle.apply(find_vbp)


    # # Find words with one of the follow pos-tags: 'DT', 'RB', 'PRP'
    # df['words_postText'] = df.processed_postText.apply(good_words)
    # df['words_targetDescription'] = df.processed_targetDescription.apply(good_words)
    # df['words_targetTitle'] = df.processed_targetTitle.apply(good_words)


    df1 =df.iloc[:,0]
    df2= df.iloc[: , 12:]
    df3 = pd.concat([df1, df2], axis=1, sort=False)
    df3.to_csv(PATH_TO_FILE, index=False)


if __name__ == '__main__':
    main()


