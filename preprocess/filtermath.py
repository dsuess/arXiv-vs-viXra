#!/usr/bin/env python
# encoding: utf-8

from pandocfilters import toJSONFilter, Str


def replace_math(key, val, fmt, meta):
    if key == 'Math':
        return Str('__equation__')

if __name__ == '__main__':
    toJSONFilter(replace_math)
