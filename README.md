# NLP hacking -- arXiv vs viXra

The [arXiv](https://arxiv.org) is a preprint server operated by Cornell University for scientific papers in physics, math, computer science, and many more.
[viXra](https://vixra.org) is -- to use the neutral description of wikipedia -- an electronic e-print archive set up by independent physicist Philip Gibbs as an alternative to the dominant arXiv service.
The motivation behind viXra is to cater to researchers, who believe that their preprints had been unfairly rejected or reclassified by the arXiv.
Here, we are trying to answer the question whether we can distinguish papers from the arXiv from papers from viXra only based on their title and abstract.

## Getting the data

The `get_data` script downloads the necessary raw data into the `data/raw/` directory.
Each article is saved in a json file of the form `arXiv_$ID.json` or `viXra_$ID.json`, where `$ID` is the id of the article.
Each file contains the following keys: id, title, abstract, and category.
Here is how it works:

- for the viXra data we run a scrapy webscraper defined in `crawlers/spiders/vixra.py`
	- it runs over all articles in the following categories: hep, qgst, relcos, astro, and quant (since these are the largest ones and they have corresponding categories on the arxiv).
	  In future versions, we can also try other categories.
	- each article is the downloaded into the given json file using the `dump_article` function in `crawlers/spiders/utils.py`.
- for the arXiv data we use the [arXiv's API](https://arxiv.org/help/api/index).
  Never crawl the actual website of the arXiv or you'll get blocked...
  - for this we run the python script `crawlers/spiders/arxiv.py`
  - to get a somewhat balanced dataset (there are many more articles on the arXiv) we only download a comparable number of articles per category.
    The also only download articles from the arXiv that have a main category corresponding to one of the viXra categories mentioned above.
    These are: hep-ph, gr-qc, astro-ph.CO, astro-ph.GA, astro-ph.HE, and quant-ph.
    For the exact number of articles see the `get_data` script.


## Preprocessing

Once the data is downloaded, we need to run the following preprocessing steps.
These might take a while (~30 min) and have not been optimised whatsoever.
Therefore, we also provide a version of the final data file `data/data.json` in the repository using git-lfs.
Run the following steps in that order:

- `python preprocess/build_vocab.py`: Generates the corpus' complete vocabulary, which is saved in `data/vocab.json`.
  It contains all unique, lower-cased word stems appearing in the dataset with their counts.
  - All LaTeX inline math is converted to single token `__equation__` using a custom pandoc filter (see `preprocess/filtermath.py`).
    The double underscore is necessary to keep the nltk stemmatizer from converting it to something equivalent to the english word "equation".
- `python preprocess/build_data.py`: Generates the data file `data/data.json` used for the actual learning, keys `data` and `vocab`
  - `'vocab'` is the actual vocabulary used for tokenizing the data (since we drop all words with less than 3 occurrences by default)
  - `'data'` contains a dictionary of articles, where the key is the article id and the value is another dictionary explained below
- each article in the final data corresponds to a dictionary with the following keys:
  - `'label'`: whether its from the arXiv or not (`true` or `false`)
  - `'text'`: list of tokens corresponding to the tokenised abstract

## Models

So far, we have only implemented a simple convolutional classifier in `models/conv.py`, which nevertheless works pretty well.
It was inspired by the [deep-scite](https://github.com/silverpond/deep-scite) model.
After running `python models/conv.py`, which saves the model under `/tmp/tf-checkpoints/words` regularly, you can evaluate it on a test set using the `arXiv vs viXra.ipynb` jupyter notebook.
There, we also show how to test it on abstracts provided by the user.

## TODO

- provide separate train/test data sets -- that's quite dodgy at the moment
- implement other models
- find out why the convolutional model classifies a given abstract; it's quite simple here because it basically only a score over ngrams...
