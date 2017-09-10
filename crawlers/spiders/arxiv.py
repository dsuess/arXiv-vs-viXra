import logging as logger

import click
import colorama
import feedparser

from crawlers.spiders.utils import ArticleData, dump_article
from tqdm import tqdm

colorama.init()


query_url = 'http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=lastUpdatedDate&max_results={entries}'
@click.command()
@click.argument('category', default='quant-ph', type=str)
@click.argument('entries', default=1, type=int)
@click.argument('datadir', default='data/',
                type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
def crawl(category, entries, datadir):
    query = query_url.format(category=category, entries=entries)
    feed = feedparser.parse(query)

    for article in tqdm(feed['entries']):
        article_id = article['id'].split('/')[-1].split('v')[0]
        primary_category = article['arxiv_primary_category']['term']

        if primary_category != category:
            logger.debug(colorama.RED + 'Categories do not match for ' + article_id)

        data = ArticleData(id=f'arXiv:{article_id}', title=article['title'],
                           abstract=article['summary'], category=primary_category)
        dump_article(data, datadir)


if __name__ == '__main__':
    crawl()
