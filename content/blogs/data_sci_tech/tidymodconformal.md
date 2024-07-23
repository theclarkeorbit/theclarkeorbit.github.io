---
title: "Tidymodels and conformal prediction"
Date: 22 Jul 2024
output:
  html_document:
    df_print: paged
---




### Reading material:

1.  [The tidy modeling book](https://www.tmwr.org/)
2.  [The tidymodels blog on conformal regression](https://www.tidymodels.org/learn/models/conformal-regression/)
3.  [The notes of Angelopoulos](https://arxiv.org/abs/2107.07511)
4.  [The notes of Tibshirani](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)
5.  [The book of Christoph Molnar](https://christophmolnar.com/books/conformal-prediction/)
6.  [The book of Valeriy Manokhin](https://maven.com/valeriy-manokhin/applied-conformal-prediction)
7.  [The package](https://github.com/herbps10/AdaptiveConformal) of [Sussman et al.](https://arxiv.org/abs/2312.00448)

### Getting some data

We will look at [Indian trade data](https://www.kaggle.com/datasets/lakshyaag/india-trade-data) hosted on Kaggle for the purposes of illustrating the tidy modeling techniques, without focussing too much on exploring the data.


``` bash
<!-- pip install kaggle -->
<!-- mkdir ~/.datasets/india_trade_data -->
<!-- kaggle datasets download -d lakshyaag/india-trade-data -->
<!-- mv india-trade-data.zip ~/.datasets/india_trade_data/ -->
<!-- unzip -d ~/.datasets/india_trade_data ~/.datasets/india_trade_data/india-trade-data.zip  -->
```

```
## bash: -c: line 1: syntax error near unexpected token `newline'
## bash: -c: line 1: `<!-- pip install kaggle -->'
```
Let us glance at the 4 files of data we have here:



Just for simplicity, we will stick to the HS2 files (2010-2021), and ignore the other two (HS trade data) files that run from 2010-2018. We will combine the two files into a single dataframe with an added column indicating direction of trade. We also interpret NAs in the `value` column to mean that there was no trade, and replace those with 0.


``` r
df_hs2_exp |> mutate(trade_direction = "export") -> df_hs2_exp
df_hs2_imp |> mutate(trade_direction = "import") -> df_hs2_imp

df_hs2 <- bind_rows(list(df_hs2_exp, df_hs2_imp)) |> 
  mutate(value = case_when(
    is.na(value) ~ 0,
    TRUE ~ value
  ),
  year = as.integer(year),
  country = case_when(
    country == "U S A" ~ "United states of America",
    country == "U K" ~ "United Kingdom",
    TRUE ~ country
  ))

df_hs2 |> sample_n(5)
```

```
## # A tibble: 5 × 6
##   HSCode Commodity                           value country  year trade_direction
##   <chr>  <chr>                               <dbl> <chr>   <int> <chr>          
## 1 55     MAN-MADE STAPLE FIBRES.              5.22 HONG K…  2015 export         
## 2 03     FISH AND CRUSTACEANS, MOLLUSCS AND… 36.0  KUWAIT   2015 export         
## 3 48     PAPER AND PAPERBOARD; ARTICLES OF …  0.13 TANZAN…  2010 import         
## 4 13     LAC; GUMS, RESINS AND OTHER VEGETA…  0.31 ISRAEL   2020 import         
## 5 30     PHARMACEUTICAL PRODUCTS              0    JAMAICA  2021 import
```

To make this something of a modelling challenge, we will enhance the data with some information about the countries India is trading with, like the GDP. We downloaded GDP data from the World Bank with the `wbstats` package.


``` r
indicators <- c("gdp" = "NY.GDP.MKTP.CD", # GDP in current dollars
                "population" = "SP.POP.TOTL", # population
                "land_area" = "EN.LAND.TOTL" # total land
                )
WDI(indicator = indicators, start = 2010, end = 2021) -> wb_df
```

```
## Warning in WDI(indicator = indicators, start = 2010, end = 2021): The following indicators could not be downloaded: EN.LAND.TOTL.
## 
## Please make sure that you are running the latest version of the `WDI` package, and that the arguments you are using in the `WDI()` function are valid.
## 
## Sometimes, downloads will suddenly stop working, even if nothing has changed in the R code of the WDI package. ("The same WDI package version worked yesterday!") In those cases, the problem is almost certainly related to the World Bank servers or to your internet connection.
## 
## You can check if the World Bank web API is currently serving the indicator(s) of interest by typing a URL of this form in your web browser:
## 
## https://api.worldbank.org/v2/en/country/all/indicator/EN.LAND.TOTL?format=json&date=:&per_page=32500&page=1
```

``` r
wb_df |> sample_n(10)
```

```
##                                       country iso2c iso3c year          gdp
## 1                                       China    CN   CHN 2017 1.231049e+13
## 2                                      France    FR   FRA 2012 2.683672e+12
## 3                    St. Martin (French part)    MF   MAF 2011 7.758757e+08
## 4                  Middle East & North Africa    ZQ   MEA 2014 3.627618e+12
## 5                                     Lesotho    LS   LSO 2016 2.114426e+09
## 6                          Sub-Saharan Africa    ZG   SSF 2019 1.830461e+12
## 7                                      Uganda    UG   UGA 2018 3.292703e+10
## 8                                    Cambodia    KH   KHM 2015 1.804995e+10
## 9                                       Samoa    WS   WSM 2018 8.784484e+08
## 10 East Asia & Pacific (IDA & IBRD countries)    T4   TEA 2014 1.278447e+13
##    population
## 1  1396215000
## 2    65662240
## 3       36350
## 4   431664579
## 5     2143872
## 6  1121549049
## 7    41515395
## 8    15417523
## 9      209701
## 10 2009149689
```

Now, we will do a left join onto the Indian trade data on country and yeay, by first converting the countries in the Indian trade to their ISO3 codes via the `countries` package.


``` r
df_hs2 |> mutate(iso3c = country_name(x = country, to="ISO3")) -> df_hs2
```

```
## Some country IDs have no match in one or more of the requested country naming conventions, NA returned.
```

```
## Multiple country IDs have been matched to the same country name.
```

```
## There is low confidence on the matching of some country names, NA returned.
```

```
## 
## Set - verbose - to TRUE for more details
```

``` r
df_hs2[is.na(df_hs2$iso3c),]$country |> unique()
```

```
##  [1] "C AFRI REP"                   "CANARY IS"                   
##  [3] "FR GUIANA"                    "FR POLYNESIA"                
##  [5] "FR S ANT TR"                  "N. MARIANA IS."              
##  [7] "NETHERLANDANTIL"              "PACIFIC IS"                  
##  [9] "UNION OF SERBIA & MONTENEGRO" "UNSPECIFIED"                 
## [11] "CHANNEL IS"                   "NEUTRAL ZONE"                
## [13] "US MINOR OUTLYING ISLANDS"
```
 There are a few countries for which we could not find the ISO3 codes, but these seem minor, so no matter. 
 
 Now, we construct our modelling data frame, where we will try to predict Indias export and imports to a given country based on somebasic characteristics. 

``` r
df <- left_join(df_hs2, wb_df, by = c("iso3c", "year")) |> 
  select(Commodity, value, country = country.y, year, trade_direction, iso3c, gdp, population) |> 
  group_by(country, year, trade_direction) |> 
  summarise(value = sum(value), gdp = gdp[1], population = population[1], .groups = "drop") |> 
  select(-country)

df |> sample_n(10)
```

```
## # A tibble: 10 × 5
##     year trade_direction    value      gdp population
##    <int> <chr>              <dbl>    <dbl>      <dbl>
##  1  2011 import          11999.    6.23e12  127833000
##  2  2010 import            125.    3.71e10    3097282
##  3  2016 import            908.    3.96e11    8736668
##  4  2017 export              7.41 NA          3396933
##  5  2012 import              0     3.93e 7      10854
##  6  2011 import             40.9   1.22e10   12317730
##  7  2021 export          11150.    4.34e11    5453566
##  8  2017 import              9.65  9.17e 9     270810
##  9  2010 import              0.08  6.87e 8      29726
## 10  2020 import          12773.    1.64e12   51836239
```

``` r
rm(df_exp, df_hs2, df_hs2_exp, df_hs2_imp, df_imp, wb_df)
```
 
### Tidy modeling in R

