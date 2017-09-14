from glob import glob

import nltk
import numpy as np
import simplejson
from preprocess.build_vocab import generate_tokens, load_dataset
from tqdm import tqdm


def load_vocab(vocabpath, min_occurences=3):
    with open(vocabpath, 'r') as buf:
        vocab = simplejson.load(buf)

    keys = np.asarray(list(vocab.keys()))
    counts = np.fromiter(vocab.values(), np.int32)
    sel = counts > min_occurences
    keys, counts = keys[sel], counts[sel]

    order = np.argsort(counts)[::-1]
    return {key: n for n, key in enumerate(keys[order])}


def encode_text(raw_text, vocab_idx, stemmatize=None):
    if stemmatize is None:
        stemmatizer = nltk.stem.PorterStemmer()
        stemmatize = stemmatizer.stem

    try:
        tokens = generate_tokens(raw_text, stemmatize)
        encoded = [vocab_idx[token] for token in tokens if token in vocab_idx]
    except RuntimeError as e:
        encoded = None
    return encoded



def main(globpath='data/raw/*.json', vocabpath='data/vocab.json',
         outfile='data/data.json', min_occurences=3):
    vocab_idx = load_vocab(vocabpath, min_occurences)
    datasets = (load_dataset(filename) for filename in tqdm(glob(globpath)))
    result = {data['id']: {'label': data['id'].startswith('arXiv'),
                           'text': encode_text(data['abstract'], vocab_idx)}
              for data in datasets}

    with open(outfile, 'w') as buf:
        simplejson.dump({'data': result, 'vocab': vocab_idx}, buf)


if __name__ == '__main__':
    main('data/raw/*.json')
