import os

import scrapy
import simplejson


class BreitbartSpider(scrapy.Spider):
    name = 'breitbart'
    baseurl = 'http://www.breitbart.com'
    article_counter = 0

    def __init__(self, topic='big-government', start_page=1, end_page=1,
                 fileprefix='data/breitbart'):
        self.topic = topic
        self.start_page = int(start_page)
        self.end_page = int(end_page)
        self.fileprefix = fileprefix

    def start_requests(self):
        for page in range(self.start_page, self.end_page + 1):
            url = f'{self.baseurl}/{self.topic}/page/{page}'
            self.logger.info(f'Scraped {self.article_counter} articles so far')
            yield scrapy.Request(url=url, callback=self.parse_index)

    def parse_index(self, response):
        articles = response.css('div.article-list a.thumbnail-url::attr(href)')
        nr_articles = len(articles)
        self.logger.info(f'Found {nr_articles} articles in {response.url}')

        for article in articles:
            yield scrapy.Request(url=article.extract(), callback=self.parse_article)

    def parse_article(self, response):
        self.logger.debug(f'Parsing article {response.url}')

        main_elements = response.xpath('//div[@class="entry-content"]/p')
        body = ' '.join(main_elements.css('::text').extract())
        data = dict(
            header=response.css('article.post h1::text').extract_first(),
            abstract=response.css('div.entry-content h2::text').extract_first(),
            body=body, url=response.url, id=self.article_counter)

        outfile_name = f'{self.fileprefix}_{self.article_counter}.json'
        with open(outfile_name, 'w', encoding='utf-8') as buf:
            simplejson.dump(data, buf)
        BreitbartSpider.article_counter += 1



