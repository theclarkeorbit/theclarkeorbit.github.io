---
title: "Tidymodels and conformal prediction"
Date: 22 Jul 2024
output:
  html_document:
    df_print: paged
---


``` r
library(tidymodels)
```

```
## ── Attaching packages ────────────────────────────────────── tidymodels 1.2.0 ──
```

```
## ✔ broom        1.0.6     ✔ recipes      1.1.0
## ✔ dials        1.2.1     ✔ rsample      1.2.1
## ✔ dplyr        1.1.4     ✔ tibble       3.2.1
## ✔ ggplot2      3.5.1     ✔ tidyr        1.3.1
## ✔ infer        1.0.7     ✔ tune         1.2.1
## ✔ modeldata    1.4.0     ✔ workflows    1.1.4
## ✔ parsnip      1.2.1     ✔ workflowsets 1.1.0
## ✔ purrr        1.0.2     ✔ yardstick    1.3.1
```

```
## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
## ✖ purrr::discard() masks scales::discard()
## ✖ dplyr::filter()  masks stats::filter()
## ✖ dplyr::lag()     masks stats::lag()
## ✖ recipes::step()  masks stats::step()
## • Search for functions across packages at https://www.tidymodels.org/find/
```

``` r
library(WDI)
library(tidyverse)
```

```
## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
## ✔ forcats   1.0.0     ✔ readr     2.1.5
## ✔ lubridate 1.9.3     ✔ stringr   1.5.1
```

```
## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ readr::col_factor() masks scales::col_factor()
## ✖ purrr::discard()    masks scales::discard()
## ✖ dplyr::filter()     masks stats::filter()
## ✖ stringr::fixed()    masks recipes::fixed()
## ✖ dplyr::lag()        masks stats::lag()
## ✖ readr::spec()       masks yardstick::spec()
## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
```

``` r
library(ggthemes)
library(xgboost)
```

```
## 
## Attaching package: 'xgboost'
## 
## The following object is masked from 'package:dplyr':
## 
##     slice
```

``` r
library(countries)
```


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


``` r
read_csv("~/.datasets/india_trade_data/2010_2021_HS2_export.csv") -> df_hs2_exp
```

```
## Rows: 184755 Columns: 5
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr (3): HSCode, Commodity, country
## dbl (2): value, year
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
read_csv("~/.datasets/india_trade_data/2010_2021_HS2_import.csv") -> df_hs2_imp
```

```
## Rows: 101051 Columns: 5
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr (3): HSCode, Commodity, country
## dbl (2): value, year
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
read_csv("~/.datasets/india_trade_data/2018-2010_export.csv") -> df_exp
```

```
## Rows: 137023 Columns: 5
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr (2): Commodity, country
## dbl (3): HSCode, value, year
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
read_csv("~/.datasets/india_trade_data/2018-2010_import.csv") -> df_imp
```

```
## Rows: 76124 Columns: 5
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr (2): Commodity, country
## dbl (3): HSCode, value, year
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
df_hs2_exp |> head()
```

```
## # A tibble: 6 × 5
##   HSCode Commodity                                           value country  year
##   <chr>  <chr>                                               <dbl> <chr>   <dbl>
## 1 02     MEAT AND EDIBLE MEAT OFFAL.                          1.4  AFGHAN…  2010
## 2 03     FISH AND CRUSTACEANS, MOLLUSCS AND OTHER AQUATIC I…  0.08 AFGHAN…  2010
## 3 04     DAIRY PRODUCE; BIRDS' EGGS; NATURAL HONEY; EDIBLE …  3.89 AFGHAN…  2010
## 4 05     PRODUCTS OF ANIMAL ORIGIN, NOT ELSEWHERE SPECIFIED… NA    AFGHAN…  2010
## 5 06     LIVE TREES AND OTHER PLANTS; BULBS; ROOTS AND THE … NA    AFGHAN…  2010
## 6 07     EDIBLE VEGETABLES AND CERTAIN ROOTS AND TUBERS.      0.17 AFGHAN…  2010
```

``` r
df_hs2_imp |> head()
```

```
## # A tibble: 6 × 5
##   HSCode Commodity                                           value country  year
##   <chr>  <chr>                                               <dbl> <chr>   <dbl>
## 1 07     EDIBLE VEGETABLES AND CERTAIN ROOTS AND TUBERS.      9.14 AFGHAN…  2010
## 2 08     EDIBLE FRUIT AND NUTS; PEEL OR CITRUS FRUIT OR MEL… 93.8  AFGHAN…  2010
## 3 09     COFFEE, TEA, MATE AND SPICES.                        2.54 AFGHAN…  2010
## 4 12     OIL SEEDS AND OLEA. FRUITS; MISC. GRAINS, SEEDS AN…  0.32 AFGHAN…  2010
## 5 13     LAC; GUMS, RESINS AND OTHER VEGETABLE SAPS AND EXT… 37.7  AFGHAN…  2010
## 6 16     PREPARATIONS OF MEAT, OF FISH OR OF CRUSTACEANS, M…  0    AFGHAN…  2010
```

``` r
df_exp |> head()
```

```
## # A tibble: 6 × 5
##   HSCode Commodity                                           value country  year
##    <dbl> <chr>                                               <dbl> <chr>   <dbl>
## 1      2 MEAT AND EDIBLE MEAT OFFAL.                          0.18 AFGHAN…  2018
## 2      3 FISH AND CRUSTACEANS, MOLLUSCS AND OTHER AQUATIC I…  0    AFGHAN…  2018
## 3      4 DAIRY PRODUCE; BIRDS' EGGS; NATURAL HONEY; EDIBLE … 12.5  AFGHAN…  2018
## 4      6 LIVE TREES AND OTHER PLANTS; BULBS; ROOTS AND THE …  0    AFGHAN…  2018
## 5      7 EDIBLE VEGETABLES AND CERTAIN ROOTS AND TUBERS.      1.89 AFGHAN…  2018
## 6      8 EDIBLE FRUIT AND NUTS; PEEL OR CITRUS FRUIT OR MEL… 25.0  AFGHAN…  2018
```

``` r
df_imp |> head()
```

```
## # A tibble: 6 × 5
##   HSCode Commodity                                           value country  year
##    <dbl> <chr>                                               <dbl> <chr>   <dbl>
## 1      5 PRODUCTS OF ANIMAL ORIGIN, NOT ELSEWHERE SPECIFIE…   0    AFGHAN…  2018
## 2      7 EDIBLE VEGETABLES AND CERTAIN ROOTS AND TUBERS.     12.4  AFGHAN…  2018
## 3      8 EDIBLE FRUIT AND NUTS; PEEL OR CITRUS FRUIT OR ME… 269.   AFGHAN…  2018
## 4      9 COFFEE, TEA, MATE AND SPICES.                       35.5  AFGHAN…  2018
## 5     11 PRODUCTS OF THE MILLING INDUSTRY; MALT; STARCHES;…  NA    AFGHAN…  2018
## 6     12 OIL SEEDS AND OLEA. FRUITS; MISC. GRAINS, SEEDS A…   8.32 AFGHAN…  2018
```

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
## 1 33     ESSENTIAL OILS AND RESINOIDS; PERF… 24.3  United…  2010 import         
## 2 68     ARTICLES OF STONE, PLASTER, CEMENT…  3.09 AUSTRIA  2011 export         
## 3 57     CARPETS AND OTHER TEXTILE FLOOR CO…  0    ST LUC…  2011 export         
## 4 81     OTHER BASE METALS; CERMETS; ARTICL…  0.03 DENMARK  2016 export         
## 5 33     ESSENTIAL OILS AND RESINOIDS; PERF…  0.36 BOTSWA…  2018 export
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
##                                         country iso2c iso3c year          gdp
## 1      Fragile and conflict affected situations    F1   FCS 2016 1.490634e+12
## 2                    Early-demographic dividend    V2   EAR 2011 9.813801e+12
## 3                      Central African Republic    CF   CAF 2013 1.691544e+09
## 4  Europe & Central Asia (IDA & IBRD countries)    T7   TEC 2021 4.623633e+12
## 5                                       Bolivia    BO   BOL 2014 3.299619e+10
## 6                                     IDA blend    XH   IDB 2010 7.413120e+11
## 7                                       Jamaica    JM   JAM 2015 1.418894e+10
## 8                                       Myanmar    MM   MMR 2020 7.900611e+10
## 9                                   Yemen, Rep.    YE   YEM 2019           NA
## 10                                         Mali    ML   MLI 2016 1.402605e+10
##    population
## 1   896306261
## 2  2991403487
## 3     4802428
## 4   463035774
## 5    10916987
## 6   474626536
## 7     2794445
## 8    53423198
## 9    31546691
## 10   18700106
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
##     year trade_direction  value     gdp population
##    <int> <chr>            <dbl>   <dbl>      <dbl>
##  1  2018 import          1955.  5.25e11   44494502
##  2  2011 import             0.1 2.64e 9     101288
##  3  2015 export           487.  6.98e 9   13763906
##  4  2018 export           380.  2.12e11    4900600
##  5  2019 import            15.0 9.42e 9   15981300
##  6  2014 export          2782.  1.47e12   23475686
##  7  2016 import            21.7 6.61e10   15827690
##  8  2020 export           508.  1.21e11   36688772
##  9  2017 import            31.0 7.70e 9    6198200
## 10  2012 export           250.  4.40e10    5178337
```

``` r
rm(df_exp, df_hs2, df_hs2_exp, df_hs2_imp, df_imp, wb_df)
```
 
### Tidy modeling in R

