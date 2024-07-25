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
<!-- pip install kaggle==1.6.14 -->
<!-- mkdir ~/.datasets/india_trade_data -->
<!-- kaggle datasets download -d lakshyaag/india-trade-data -->
<!-- mv india-trade-data.zip ~/.datasets/india_trade_data/ -->
<!-- unzip -d ~/.datasets/india_trade_data ~/.datasets/india_trade_data/india-trade-data.zip -->
```

```
## bash: -c: line 1: syntax error near unexpected token `newline'
## bash: -c: line 1: `<!-- pip install kaggle==1.6.14 -->'
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
```

```
## Error in WDI(indicator = indicators, start = 2010, end = 2021): The following indicators could not be downloaded: NY.GDP.MKTP.CD, SP.POP.TOTL, EN.LAND.TOTL.
## 
## Please make sure that you are running the latest version of the `WDI` package, and that the arguments you are using in the `WDI()` function are valid.
## 
## Sometimes, downloads will suddenly stop working, even if nothing has changed in the R code of the WDI package. ("The same WDI package version worked yesterday!") In those cases, the problem is almost certainly related to the World Bank servers or to your internet connection.
## 
## You can check if the World Bank web API is currently serving the indicator(s) of interest by typing a URL of this form in your web browser:
## 
## https://api.worldbank.org/v2/en/country/all/indicator/NY.GDP.MKTP.CD?format=json&date=:&per_page=32500&page=1
```

``` r
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
```

```
## Error in eval(expr, envir, enclos): object 'wb_df' not found
```

``` r
rm(df_exp, df_hs2, df_hs2_exp, df_hs2_imp, df_imp, wb_df)
```

Just for simplicity, we will stick to the HS2 files (2010-2021), and ignore the other two (HS trade data) files that run from 2010-2018. We will combine the two files into a single data frame with an added column indicating direction of trade. We also interpret NAs in the `value` column to mean that there was no trade, and replace those with 0.

To make this something of a modelling challenge, we have enhanced the data with some information about the countries India is trading with, like the GDP. We downloaded GDP data from the World Bank with the `WDI` package. Then, we did a left join onto the Indian trade data on country and year, by first converting the countries in the Indian trade to their ISO3 codes via the `countries` package.

The data frame we now have will serve as the basis for us to explore tidy modeling and conformal prediction in R.


``` r
df |> sample_n(10)
```

```
## Error in `sample_n()`:
## ! `tbl` must be a data frame, not a function.
```

``` r
ggplot({df |> na.omit()}, aes(x = {value}, fill = trade_direction)) +
  geom_histogram(bins = 50, col = "white", alpha = 0.25, position = "identity") +
  scale_x_log10() +
  theme_tufte()
```

```
## Error in `ggplot()`:
## ! `data` cannot be a function.
## ℹ Have you misspelled the `data` argument in `ggplot()`
```

### Tidy modeling in R

Since the quantitative columns like `value`, `gdp` and `population` vary by orders of magnitude over the data, it probably makes sense to log transform them. To deal with 0 values, we add 1 to all the values, and omit rows which have any data missing. Now we split the data into training and test using built in functions from the `rsample` package, making sure that the distribution of the value column is similar in all our data splits using the strata argument. 


``` r
set.seed(1)

trade_split <- initial_validation_split({df |> mutate(year = as.factor(year), value = log(value+1))}, prop = c(0.6, 0.2), strata = value)
```

```
## Error in UseMethod("mutate"): no applicable method for 'mutate' applied to an object of class "function"
```

``` r
trade_split |> print()
```

```
## Error in eval(expr, envir, enclos): object 'trade_split' not found
```

``` r
train_df <- training(trade_split)
```

```
## Error in eval(expr, envir, enclos): object 'trade_split' not found
```

``` r
val_df <- validation(trade_split)
```

```
## Error in eval(expr, envir, enclos): object 'trade_split' not found
```

``` r
test_df <- testing(trade_split)
```

```
## Error in eval(expr, envir, enclos): object 'trade_split' not found
```

``` r
trade_simple_regression <- 
  recipe(value ~ ., data = {train_df |> 
      select(-country)}) |> 
  step_naomit() |> 
  step_log(gdp, population) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_normalize(gdp, population)
```

```
## Error in eval(expr, envir, enclos): object 'train_df' not found
```

#### Linear model

As a first baseline, always best ton begin with a simple, interpretable linear model.


``` r
linear_model <- linear_reg() |> 
  set_engine("lm") 

linear_workflow <- workflow() |>
  add_recipe(trade_simple_regression) |> 
  add_model(linear_model)
```

```
## Error in eval(expr, envir, enclos): object 'trade_simple_regression' not found
```

``` r
linear_fit <- fit(linear_workflow, train_df)
```

```
## Error in eval(expr, envir, enclos): object 'linear_workflow' not found
```

``` r
linear_fit |> tidy()
```

```
## Error in eval(expr, envir, enclos): object 'linear_fit' not found
```

Now, let us make some predictions on the validation data.


``` r
trade_reg_metrics <- metric_set(rmse, rsq, mae)
linear_val_preds <- predict(linear_fit, new_data = val_df) |> 
  bind_cols(val_df |> select(value))
```

```
## Error in eval(expr, envir, enclos): object 'linear_fit' not found
```

``` r
# linear_val_preds |> head()
trade_reg_metrics(linear_val_preds, truth = value, estimate = .pred) |> 
  transmute(metric = .metric, linear_model_validation = .estimate) -> linear_val_perf
```

```
## Error in `metric_set()`:
## ! Failed to compute `rmse()`.
## Caused by error:
## ! object 'linear_val_preds' not found
```

``` r
linear_val_perf
```

```
## Error in eval(expr, envir, enclos): object 'linear_val_perf' not found
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
```

```
## Error in eval(expr, envir, enclos): object 'trade_simple_regression' not found
```

``` r
xgb_fit <- fit(xgb_workflow, train_df)
```

```
## Error in eval(expr, envir, enclos): object 'xgb_workflow' not found
```

``` r
xgb_val_preds <- predict(xgb_fit, new_data = val_df) |> 
  bind_cols(val_df |> select(value))
```

```
## Error in eval(expr, envir, enclos): object 'xgb_fit' not found
```

``` r
trade_reg_metrics(xgb_val_preds, truth = value, estimate = .pred) |> 
  transmute(metric = .metric, xgb_model_validation = .estimate) -> xgb_val_perf
```

```
## Error in `metric_set()`:
## ! Failed to compute `rmse()`.
## Caused by error:
## ! object 'xgb_val_preds' not found
```

``` r
left_join(linear_val_perf, xgb_val_perf)
```

```
## Error in eval(expr, envir, enclos): object 'linear_val_perf' not found
```

#### Multiclass classification

Here, we will add back the country column and widen the data frame to show value of goods imported and exported by from this country, and try to predict the country based on population, GDP, and trade with India. 


``` r
spread(df, key = trade_direction, value = value) |> 
  na.omit() |> 
  mutate(country = as.factor(country)) |> 
  select(-year) -> df_classification
```

```
## Error in UseMethod("spread"): no applicable method for 'spread' applied to an object of class "function"
```

``` r
trade_class_split <- initial_split(df_classification, prop = c(0.6), strata = country)
```

```
## Error in eval(expr, envir, enclos): object 'df_classification' not found
```

``` r
class_train_df <- training(trade_class_split)
```

```
## Error in eval(expr, envir, enclos): object 'trade_class_split' not found
```

``` r
class_test_df <- testing(trade_class_split)
```

```
## Error in eval(expr, envir, enclos): object 'trade_class_split' not found
```

``` r
trade_simple_classification <- 
  recipe(country ~ ., data = class_train_df) |> 
  step_naomit() |> 
  step_log(gdp, population) |> 
  step_normalize(all_numeric_predictors())
```

```
## Error in eval(expr, envir, enclos): object 'class_train_df' not found
```

``` r
xgb_class_model <- boost_tree(mtry = 3, trees = 1000, min_n = 5, tree_depth = 5, learn_rate = 0.01) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

xgb_class_workflow <- workflow() |>
  add_recipe(trade_simple_classification) |> 
  add_model(xgb_class_model)
```

```
## Error in eval(expr, envir, enclos): object 'trade_simple_classification' not found
```

``` r
xgb_class_fit <- fit(xgb_class_workflow, class_train_df)
```

```
## Error in eval(expr, envir, enclos): object 'xgb_class_workflow' not found
```

``` r
xgb_class_test_preds <- predict(xgb_class_fit, new_data = class_test_df) |> 
  bind_cols(class_test_df |> select(country))
```

```
## Error in eval(expr, envir, enclos): object 'xgb_class_fit' not found
```

``` r
class_metrics <- metric_set(accuracy, mcc)
class_metrics(xgb_class_test_preds, truth = country, estimate = .pred_class) |> 
  transmute(metric = .metric, xgb_class_validation = .estimate) -> xgb_class_perf
```

```
## Error in `metric_set()`:
## ! Failed to compute `accuracy()`.
## Caused by error:
## ! object 'xgb_class_test_preds' not found
```

``` r
xgb_class_perf
```

```
## Error in eval(expr, envir, enclos): object 'xgb_class_perf' not found
```
 An accuracy of 60% for a messy classification problem with 208 classes is not too shabby at all, but adding any notion of confidence/coverage to a particular prediction is difficult (even if we have some kind of un-calibrated probability given by XGBoost), which is where conformal prediction comes it. 
 
### Conformal prediction

...

