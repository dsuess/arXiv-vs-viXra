import functools as ft

import colorama

import scrapy

from .utils import ArticleData, dump_article

colorama.init()


class VixraSpider(scrapy.Spider):
    name = 'vixra'
    categories = ['clim']
    baseurl = 'http://vixra.org'

    def __init__(self, datadir='data/'):
        self.datadir = datadir

    def start_requests(self):
        for category in self.categories:
            url = f'{self.baseurl}/{category}'
            callback = ft.partial(self.parse_index, category=category)
            yield scrapy.Request(url=url, callback=callback)

    def parse_index(self, response, category):
        prev_months = response.xpath('(//div[@id="flow"]/p)[1]/a/@href').extract()
        nr_hits = len(prev_months)
        self.logger.debug(f'Found {nr_hits} indices in {response.url}')

        for month in prev_months:
            url = f'{response.url}/{month}'
            callback = ft.partial(self.parse_articles, category=category)
            yield scrapy.Request(url=url, callback=callback)

    def parse_articles(self, response, category):
        ids = response.xpath('//div[@id="flow"]/p/b/a[1]/text()').extract()
        articles = response.xpath('//div[@id="flow"]/div[@id="abstract"]')
        nr_articles = len(articles)
        self.logger.debug(f'Found {nr_articles} articles in {response.url}')

        if len(ids) != len(articles):
            self.logger.error(colorama.RED + f'Could not match ids and articles')
            return

        for article_id, article in zip(ids, articles):
            title = article.css('h3::text').extract_first()
            abstract = article.xpath('p[2]/text()').extract_first().strip()
            article_data = ArticleData(id=article_id, title=title,
                                       abstract=abstract, category=category)
            dump_article(article_data, self.datadir)
