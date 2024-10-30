---
title: "Basics of time series, and adaptive conformal inference"
Date: 01 Aug 2024
output:
  html_document:
    df_print: paged
editor_options: 
  markdown: 
    wrap: 72
---



## Getting data

For this post, we will use data from a [recent kaggle contest](https://www.kaggle.com/competitions/probabilistic-forecasting-i-temperature/) on probabilistic time series prediction. With a free Kaggle API, the data can be downloaded via the cli. 


``` bash
# kaggle competitions download -c probabilistic-forecasting-i-temperature
```


``` r
train <- read_csv('probabilistic-forecasting-i-temperature/train.csv')
```

```
## Rows: 64320 Columns: 9
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## dbl  (8): id, feature_AA, feature_AB, feature_BA, feature_BB, feature_CA, fe...
## dttm (1): date
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
train |> head(5)
```

```
## # A tibble: 5 × 9
##      id date                feature_AA feature_AB feature_BA feature_BB
##   <dbl> <dttm>                   <dbl>      <dbl>      <dbl>      <dbl>
## 1     0 2016-07-01 00:00:00       5.83       2.01       1.60      0.462
## 2     1 2016-07-01 00:15:00       5.76       2.08       1.49      0.426
## 3     2 2016-07-01 00:30:00       5.76       1.94       1.49      0.391
## 4     3 2016-07-01 00:45:00       5.76       1.94       1.49      0.426
## 5     4 2016-07-01 01:00:00       5.69       2.08       1.49      0.426
## # ℹ 3 more variables: feature_CA <dbl>, feature_CB <dbl>, Temperature <dbl>
```

``` r
explore::describe(train)
```

```
## Error in loadNamespace(x): there is no package called 'explore'
```

