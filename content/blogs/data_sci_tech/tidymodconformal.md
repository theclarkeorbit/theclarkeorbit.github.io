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
##    country           year trade_direction   value     gdp population
##    <chr>            <int> <chr>             <dbl>   <dbl>      <dbl>
##  1 Zimbabwe          2021 import             7.77 2.84e10   15993524
##  2 Marshall Islands  2021 import             1.18 2.58e 8      42050
##  3 Italy             2020 export          4736.   1.90e12   59438851
##  4 Japan             2020 export          4434.   5.06e12  126261000
##  5 Ethiopia          2017 import            47.4  8.18e10  108197950
##  6 Bahamas, The      2014 export           596.   1.10e10     389131
##  7 Malawi            2016 export           178.   7.91e 9   17405624
##  8 Thailand          2011 export          2961.   3.71e11   68712846
##  9 Estonia           2014 import           118.   2.66e10    1314545
## 10 Lebanon           2018 export           269.   5.49e10    5950839
```

``` r
ggplot({df |> na.omit()}, aes(x = {value}, fill = trade_direction)) +
  geom_histogram(bins = 50, col = "white", alpha = 0.25, position = "identity") +
  scale_x_log10() +
  theme_tufte()
```

![center](/figures/tidymodconformal/unnamed-chunk-3-1.png)

### Tidy modeling in R

Since the quantitative columns like `value`, `gdp` and `population` vary by orders of magnitude over the data, it probably makes sense to log transform them. To deal with 0 values, we add 1 to all the values, and omit rows which have any data missing. Now we split the data into training and test using built in functions from the `rsample` package, making sure that the distribution of the value column is similar in all our data splits using the strata argument. 


``` r
set.seed(1)

trade_split <- initial_validation_split({df |> mutate(year = as.factor(year), value = log(value+1))}, prop = c(0.6, 0.2), strata = value)
trade_split |> print()
```

```
## <Training/Validation/Testing/Total>
## <2944/981/984/4909>
```

``` r
train_df <- training(trade_split)
val_df <- validation(trade_split)
test_df <- testing(trade_split)


trade_simple_regression <- 
  recipe(value ~ ., data = {train_df |> 
      select(-country)}) |> 
  step_naomit() |> 
  step_log(gdp, population) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_normalize(gdp, population)
```

#### Linear model

As a first baseline, always best ton begin with a simple, interpretable linear model.


``` r
linear_model <- linear_reg() |> 
  set_engine("lm") 

linear_workflow <- workflow() |>
  add_recipe(trade_simple_regression) |> 
  add_model(linear_model)

linear_fit <- fit(linear_workflow, train_df)

linear_fit |> tidy()
```

```
## # A tibble: 15 × 5
##    term                   estimate std.error statistic   p.value
##    <chr>                     <dbl>     <dbl>     <dbl>     <dbl>
##  1 (Intercept)              4.86      0.102     47.8   0        
##  2 gdp                      1.61      0.0471    34.2   3.32e-215
##  3 population               0.906     0.0475    19.1   1.39e- 76
##  4 year_X2011               0.119     0.136      0.872 3.83e-  1
##  5 year_X2012               0.155     0.138      1.12  2.64e-  1
##  6 year_X2013              -0.0318    0.138     -0.231 8.18e-  1
##  7 year_X2014               0.0517    0.137      0.378 7.06e-  1
##  8 year_X2015               0.157     0.138      1.14  2.54e-  1
##  9 year_X2016               0.0436    0.141      0.309 7.57e-  1
## 10 year_X2017               0.201     0.138      1.46  1.46e-  1
## 11 year_X2018               0.135     0.138      0.977 3.29e-  1
## 12 year_X2019               0.0980    0.137      0.715 4.75e-  1
## 13 year_X2020               0.0755    0.138      0.545 5.86e-  1
## 14 year_X2021               0.346     0.136      2.55  1.09e-  2
## 15 trade_direction_import  -0.488     0.0558    -8.75  3.46e- 18
```

Now, let us make some predictions on the validation data.


``` r
linear_fit_last <- last_fit(linear_workflow, trade_split) 
trade_reg_metrics <- metric_set(rmse, rsq, mae)
collect_predictions(linear_fit_last) -> linear_preds
trade_reg_metrics(linear_preds, truth = value, estimate = .pred)
```

```
## # A tibble: 3 × 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 rmse    standard       1.42 
## 2 rsq     standard       0.739
## 3 mae     standard       1.10
```

##### Regression with XGBoost

We will now use the same modules of the parsnip package with XGBoost, since this is more likely to be used in production usecases. Could hardly be easier.


``` r
xgb_model <- boost_tree(mtry = 3, trees = 500, min_n = 5, tree_depth = 3, learn_rate = 0.01) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

xgb_workflow <- workflow() |>
  add_recipe(trade_simple_regression) |> 
  add_model(xgb_model)

xgb_fit <- fit(xgb_workflow, train_df)

xgb_fit_last <- last_fit(xgb_workflow, trade_split) 
xgb_trade_reg_metrics <- metric_set(rmse, rsq, mae)
collect_predictions(xgb_fit_last) -> xgb_preds
xgb_trade_reg_metrics(xgb_preds, truth = value, estimate = .pred)
```

```
## # A tibble: 3 × 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 rmse    standard       1.36 
## 2 rsq     standard       0.776
## 3 mae     standard       1.06
```

#### Multiclass classification

Here, we will add back the country column and widen the data frame to show value of goods imported and exported by from this country, and try to predict the country based on population, GDP, and trade with India. 


``` r
spread(df, key = trade_direction, value = value) |> 
  na.omit() |> 
  mutate(country = as.factor(country)) |> 
  select(-year) -> df_classification

trade_class_split <- initial_split(df_classification, prop = c(0.6), strata = country)
```

```
## Warning: Too little data to stratify.
## • Resampling will be unstratified.
```

``` r
class_train_df <- training(trade_class_split)
class_test_df <- testing(trade_class_split)

trade_simple_classification <- 
  recipe(country ~ ., data = class_train_df) |> 
  step_naomit() |> 
  step_log(gdp, population) |> 
  step_normalize(all_numeric_predictors())

xgb_class_model <- boost_tree(mtry = 3, trees = 500, min_n = 5, tree_depth = 3, learn_rate = 0.01) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

xgb_class_workflow <- workflow() |>
  add_recipe(trade_simple_classification) |> 
  add_model(xgb_class_model)

xgb_class_fit <- fit(xgb_class_workflow, class_train_df)

xgb_class_fit_last <- last_fit(xgb_class_workflow, trade_class_split)
```

```
## → A | warning: ✖ No observations were detected in `truth` for levels: China and Eritrea.
##                ℹ Computation will proceed by ignoring those levels.
```

```
## There were issues with some computations   A: x1
## There were issues with some computations   A: x1
```

```
## 
```

``` r
collect_predictions(xgb_class_fit_last) -> xgb_class_preds
collect_metrics(xgb_class_fit_last)
```

```
## # A tibble: 3 × 4
##   .metric     .estimator .estimate .config             
##   <chr>       <chr>          <dbl> <chr>               
## 1 accuracy    multiclass     0.521 Preprocessor1_Model1
## 2 roc_auc     hand_till      0.977 Preprocessor1_Model1
## 3 brier_class multiclass     0.436 Preprocessor1_Model1
```

### Conformal prediction

...

