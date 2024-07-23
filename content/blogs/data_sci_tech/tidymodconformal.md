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

We will look at [Indian trade data](https://www.kaggle.com/datasets/lakshyaag/india-trade-data) hosted on Kaggle for the purposes of illustrating the tidy modeling techniques, without focusing too much on exploring the data.


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

indicators <- c("gdp" = "NY.GDP.MKTP.CD", # GDP in current dollars
                "population" = "SP.POP.TOTL", # population
                "land_area" = "EN.LAND.TOTL" # total land
                )
WDI(indicator = indicators, start = 2010, end = 2021) -> wb_df

df_hs2 |> mutate(iso3c = country_name(x = country, to="ISO3")) -> df_hs2
```

```
## Some country IDs have no match in one or more of the requested country naming conventions, NA returned.
## Multiple country IDs have been matched to the same country name.
## There is low confidence on the matching of some country names, NA returned.
## 
## Set - verbose - to TRUE for more details
```

``` r
df <- left_join(df_hs2, wb_df, by = c("iso3c", "year")) |> 
  select(Commodity, value, country = country.y, year, trade_direction, iso3c, gdp, population) |> 
  group_by(country, year, trade_direction) |> 
  summarise(value = sum(value), gdp = gdp[1], population = population[1], .groups = "drop")

rm(df_exp, df_hs2, df_hs2_exp, df_hs2_imp, df_imp, wb_df)
```

Just for simplicity, we will stick to the HS2 files (2010-2021), and ignore the other two (HS trade data) files that run from 2010-2018. We will combine the two files into a single data frame with an added column indicating direction of trade. We also interpret NAs in the `value` column to mean that there was no trade, and replace those with 0.

To make this something of a modelling challenge, we have enhanced the data with some information about the countries India is trading with, like the GDP. We downloaded GDP data from the World Bank with the `WDI` package. Then, we did a left join onto the Indian trade data on country and year, by first converting the countries in the Indian trade to their ISO3 codes via the `countries` package.

The data frame we now have will serve as the basis for us to explore tidy modeling and conformal prediction in R.


``` r
df |> sample_n(10)
```

```
## # A tibble: 10 × 6
##    country              year trade_direction   value           gdp population
##    <chr>               <int> <chr>             <dbl>         <dbl>      <dbl>
##  1 Burundi              2019 import             3.82   2576518880.   11874838
##  2 Trinidad and Tobago  2018 export            83.7   24569079488.    1504709
##  3 Belarus              2016 export            40.2   47723545321.    9469379
##  4 Romania              2020 export           372.   251362514350.   19265250
##  5 Bulgaria             2020 export           170.    70368758395.    6934015
##  6 North Macedonia      2019 export            22.6   12606338449.    1876262
##  7 Timor-Leste          2016 export             2.31   1652603700     1224562
##  8 Singapore            2013 import          6762.   307576360585.    5399162
##  9 Gambia, The          2010 import            14.7    1543294927.    1937275
## 10 Cote d'Ivoire        2011 import           466.    36693710801.   21562914
```

``` r
ggplot({df |> na.omit()}, aes(x = {value}, fill = trade_direction)) +
  geom_histogram(bins = 50, col = "white", alpha = 0.25, position = "identity") +
  scale_x_log10() +
  theme_tufte()
```

![center](/figures/tidymodconformal/unnamed-chunk-3-1.png)

### Tidy modeling in R

Since the quantitative columns like `value`, `gdp` and `population` vary by orders of magnitude over the data, it probably makes sense to log transform them. To deal with 0 values, we add 1 to all the values, and omit rows which have any data missing.


``` r
df |> 
  na.omit() |> 
  mutate(value = log(value+1), gdp = log(gdp+1), population = log(population+1)) -> df_modeling
```

Now we split the data into training and test using built in functions from the `rsample` package, making sure that the distribution of the value column is similar in all our data splits using the strata argument. 


``` r
set.seed(1)

trade_split <- initial_validation_split({df_modeling |> 
    mutate(year = as.numeric(year)) |> 
    select(-country)}, 
    prop = c(0.6, 0.2), strata = value)
trade_split |> print()
```

```
## <Training/Validation/Testing/Total>
## <2856/952/952/4760>
```

``` r
train_df <- training(trade_split)
val_df <- validation(trade_split)
test_df <- testing(trade_split)
```

#### Linear model

As a first baseline, always best ton begin with a simple, interpretable linear model.


``` r
linear_fit <- linear_reg() |> 
  set_engine("lm") |> 
  fit(value ~ ., data = train_df)

linear_fit |> tidy()
```

```
## # A tibble: 5 × 5
##   term                  estimate std.error statistic   p.value
##   <chr>                    <dbl>     <dbl>     <dbl>     <dbl>
## 1 (Intercept)           -47.8     16.0         -2.98 2.91e-  3
## 2 year                    0.0152   0.00796      1.91 5.66e-  2
## 3 trade_directionimport  -0.469    0.0545      -8.62 1.11e- 17
## 4 gdp                     0.666    0.0195      34.1  2.07e-214
## 5 population              0.396    0.0198      20.0  1.09e- 83
```

Now, let us make some predictions on the validation data.


``` r
val_df |> 
  select(value) |> 
  bind_cols(predict(linear_fit, val_df)) |> 
  bind_cols(predict(linear_fit, val_df, type = "pred_int")) |> # 95% prediction intervals
  mutate(linear_preds = .pred, 
         linear_pred_lower = .pred_lower,
         linear_pred_upper = .pred_upper) |> 
  select(-.pred, -.pred_lower, -.pred_upper) |> 
  mutate(linear_residuals = value - linear_preds) |> 
  bind_cols({val_df |> select(-value)}) -> results_df

ggplot(results_df) +
  geom_point(aes(x = value, y = linear_preds, colour = trade_direction)) +
  geom_smooth(aes(x = value, y = linear_preds, colour = trade_direction), method="lm") +
  theme_tufte()
```

```
## `geom_smooth()` using formula = 'y ~ x'
```

![center](/figures/tidymodconformal/unnamed-chunk-7-1.png)

