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

For this post, we will use data from a [recent kaggle contest](https://www.kaggle.com/competitions/probabilistic-forecasting-i-temperature/) on probabilistic time series prediction. With a free Kaggle API, the data can be downloaded via the cli with `kaggle competitions download -c probabilistic-forecasting-i-temperature`. 

    

