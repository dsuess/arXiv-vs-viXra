import logging as logger

import click
from colorama import init, Fore
import feedparser

from crawlers.spiders.utils import ArticleData, dump_article
from tqdm import tqdm

init()


query_url = 'http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=lastUpdatedDate&max_results={entries}'
@click.command()
@click.option('--category', default='quant-ph', type=str)
@click.option('--entries', default=1, type=int)
@click.option('--datadir', default='data/',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
def crawl(category, entries, datadir):
    print()
    print(f'Crawling {category}')
    query = query_url.format(category=category, entries=entries)
    feed = feedparser.parse(query)

    for article in tqdm(feed['entries']):
        article_id = article['id'].split('/')[-1].split('v')[0]
        primary_category = article['arxiv_primary_category']['term']

        if primary_category != category:
            logger.debug(Fore.RED + 'Categories do not match for ' + article_id + Fore.RESET)


        data = ArticleData(id=f'arXiv:{article_id}', title=article['title'],
                           abstract=article['summary'], category=primary_category)
        dump_article(data, datadir)


if __name__ == '__main__':
    crawl()
