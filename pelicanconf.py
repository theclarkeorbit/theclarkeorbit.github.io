#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'pras'
SITENAME = u'p. bhogale'
SITETITLE = u'prasanna bhogale'
SITESUBTITLE ='Consulting scientist'# \n
SITEDESCRIPTION = u'%s\'s home on the interwebz' % AUTHOR
BROWSER_COLOR = '#333333'
PYGMENTS_STYLE = 'monokai'
SITEURL = 'https://www.theclarkeorbit.com'
PATH = 'content'
STATIC_PATHS = ['images', 'pdfs', 'articles', 'tuts','blogs']
TIMEZONE = 'Europe/Paris'
DEFAULT_LANG = u'en'
DISPLAY_PAGES_ON_MENU = True
DISPLAY_CATEGORIES_ON_MENU = True
MARKDOWN = ['codehilite(css_class=highlight)','extra']
TYPOGRIFY = True
DIRECT_TEMPLATES = ['index', 'categories', 'authors', 'archives','tags']
PAGINATED_DIRECT_TEMPLATES = ['index']
SUMMARY_MAX_LENGTH = 100
WITH_FUTURE_DATES = True
SLUGIFY_SOURCE = 'title'
MONTH_ARCHIVE_SAVE_AS = 'posts/{date:%Y}/{date:%b}/index.html'
MAIN_MENU = True
MENUITEMS = (('Archives', '/archives.html'),
             ('Categories', '/categories.html'),
             ('Tags', '/tags.html'),)

SITELOGO = u'https://en.gravatar.com/userimage/9352950/78ed70e67418f76f23b494458d53ac7d.jpg?size=400'

FAVICON = SITEURL + "/images/favicon.png"

#THEME_COLOR = 'red'
#SIDEBAR_DISPLAY = ['about', 'categories']
#SIDEBAR_ABOUT = "Lorem ipsum dolor sit amet, consectetur adipisicing elit. Sequi quae hic dicta eius ad quam eligendi minima praesentium voluptatum? Quidem quaerat eaque libero velit impedit dicta, repudiandae sapiente. Deserunt, excepturi."
#DISQUS_SITENAME = 'theclarkeorbit'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None
DEFAULT_DATE='fs'
#THEME = '/usr/lib/python2.7/dist-packages/pelican/themes/svbtle'
#THEME = "pelican-mockingbird"
#THEME = "notmyidea"
#THEME = "Flex"
#THEME = "pelican-mockingbird"
THEME = "bulrush"

MARKUP = 'md'
#TYPOGRIFY = False

# Blogroll
#LINKS = (('XKCD', 'http://xkcd.com/'),
#         ('SAGEmath', 'http://www.sagemath.org/'),
#         ('The War Nerd', 'https://pando.com/author/garybrecher/'),
#         ('lib','http://gen.lib.rus.ec/'),
#         ('Peter Hitchens','http://hitchensblog.mailonsunday.co.uk/'))

# Social widget
#SOCIAL = (('You can add links in your config file', '#'),
#          ('Another social link', '#'),)
SOCIAL = (('linkedin', 'https://www.linkedin.com/in/pbhogale'),
          ('twitter','https://twitter.com/thegymnosophist'),
          ('google','https://plus.google.com/+PrasannaBhogale'),
          ('github', 'https://github.com/pbhogale'))


DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True
USE_LESS = True
