import os
from collections import namedtuple

import simplejson

ArticleData = namedtuple('ArticleData', ['id', 'title', 'abstract', 'category'])


def dump_article(data, datadir):
    assert isinstance(data, ArticleData)
    filename = data.id.replace(':', '_') + '.json'
    path = os.path.join(datadir, filename)
    with open(path, 'w') as buf:
        simplejson.dump(dict(data._asdict()), buf)
