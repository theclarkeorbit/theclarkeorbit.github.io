#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
#import render_math

#LOAD_CONTENT_CACHE = False
AUTHOR = u'pras'
SITENAME = u'p. bhogale'
SITETITLE = u'prasanna bhogale'
SITESUBTITLE ='Data Sci, Quant Fin, Quant Bio.'# \n
SITEDESCRIPTION = u'%s\'s home on the interwebz' % AUTHOR
#BROWSER_COLOR = '#333333'
PYGMENTS_STYLE = 'monokai' 
PLUGIN_PATHS = ['/']
PLUGINS = ['render_math','i18n_subsites']
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}
SITEURL = 'https://theclarkeorbit.github.io'
PATH = 'content'
STATIC_PATHS = ['images', 'pdfs', 'figures', 'pages']
TIMEZONE = 'Europe/Paris'
DEFAULT_LANG = u'en'
DISPLAY_PAGES_ON_MENU = True
USE_FOLDER_AS_CATEGORY = True
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}
MAIN_MENU = True
I18N_TEMPLATES_LANG = 'en'
#MENUITEMS = (('About', '/About.html'),)


THEME = 'pelican-bootstrap3'
GITHUB_USER = 'pbhogale'
SITELOGO = "/images/gravtar.jpg"# SITEURL +  u'https://en.gravatar.com/userimage/9352950/78ed70e67418f76f23b494458d53ac7d.jpg?size=400'
DISPLAY_RECENT_POSTS_ON_SIDEBAR = True
BANNER_SUBTITLE = 'Data Sci, Quant Fin, Quant Bio.'
FAVICON = "/images/gravtar.jpg"#"/images/favicon.png" # SITEURL + 
BANNER = '/images/banner.jpg'
BANNER_ALL_PAGES = True

# Feed generation is usually not desired when developing

DEFAULT_DATE='fs'

SOCIAL = (('linkedin', 'https://www.linkedin.com/in/pbhogale'),
          ('twitter','https://twitter.com/thegymnosophist'),
          #('reddit','https://www.reddit.com/user/thegymnosophist'),
          ('github', 'https://github.com/pbhogale'))

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True
GOOGLE_ANALYTICS = "UA-115756026-1" 



#JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}
#THEME_COLOR = 'blue'
#SIDEBAR_DISPLAY = ['About']
#SIDEBAR_ABOUT = u"हज़ारों ख़्वाहिशें ऐसी के हर ख़्वाहिश पे दम निकले \n बहुत निकले मेरे अरमाँ लेकिन फिर भी कम निकले"
#DISQUS_SITENAME = 'theclarkeorbit'
#THEME = '/usr/lib/python2.7/dist-packages/pelican/themes/svbtle'
#THEME = "pelican-mockingbird"
#THEME = "notmyidea"
#THEME = "Flex"
#THEME = "pelican-mockingbird"
#THEME = "elegant"
#DEFAULT_PAGINATION = 10
#USE_LESS = True
#ROBOTS = 'index, follow'
#TYPOGRIFY = True
#DIRECT_TEMPLATES = ['index', 'categories', 'archives','tags']
#PAGINATED_DIRECT_TEMPLATES = ['index']
#SUMMARY_MAX_LENGTH = 200
#WITH_FUTURE_DATES = True
#SLUGIFY_SOURCE = 'title'
#MONTH_ARCHIVE_SAVE_AS = 'posts/{date:%Y}/{date:%b}/index.html'
# Blogroll
#LINKS = (('XKCD', 'http://xkcd.com/'),
#         ('SAGEmath', 'http://www.sagemath.org/'),
#         ('The War Nerd', 'https://pando.com/author/garybrecher/'),
#         ('lib','http://gen.lib.rus.ec/'),
#         ('Peter Hitchens','http://hitchensblog.mailonsunday.co.uk/'))
# Social widget
#SOCIAL = (('You can add links in your config file', '#'),
#          ('Another social link', '#'),)
