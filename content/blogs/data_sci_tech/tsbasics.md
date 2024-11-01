---
title: "Basics of conformal prediction in time series data"
Date: 30 October 2024
output:
  html_document:
    df_print: paged
editor_options: 
  markdown: 
    wrap: 72
---




**Reading material**  

1. Please read the [previous blogpost](https://theclarkeorbit.github.io/tidymodels-and-conformal-prediction.html) for basics of conformal inference and more reading material.
2. This post makes use of the [`conformalForecast`](https://github.com/xqnwang/conformalForecast/) package introduced in the [paper](https://robjhyndman.com/publications/cpts.html) published just 10 days ago by Wang and Hyndman.
3. For a more general overview of time series forecasting, please see this [excellent book](https://otexts.com/fpp3/) by Hyndman.
4. [Frank Hull's](https://x.com/frankiethull) [R code](https://github.com/frankiethull/Practical-Guide-to-Applied-Conformal-Prediction-in-R) for [Valeriy Manokhin's](https://x.com/predict_addict) [book](https://www.amazon.in/dp/1805122762?ref_=cm_sw_r_cp_ud_dp_H765W8M42WJ6XJVNZGD7_1). 


Conformal prediction is a powerful, non-parametric approach to constructing prediction intervals that provide reliable coverage without requiring strong assumptions about the data distribution (see the [previous post](https://theclarkeorbit.github.io/tidymodels-and-conformal-prediction.html) for an introduction and reading material). For time series forecasting, where data is temporally correlated, conformal prediction methods must adapt to these correlations to achieve accurate and reliable intervals.

We will first generate time series data and illustrate a couple of classical forecasting techniques before moving to conformal methods.

### Generating simple time series data

Let's create a synthetic dataset that mimics real-world electricity demand and temperature patterns, where demand is influenced by temperature in a U-shaped relationship (higher demand at both very low and very high temperatures):


``` r
set.seed(2024)
n <- 52*10  # 10 years of weekly data
time <- 1:n
holdout_prop <- 0.8
level_conf <- 0.9

# Generate synthetic temperature data with seasonal pattern
temperature <- 20 + 10 * sin(2 * pi * time / 52) + rnorm(n, sd = 1)

# Generate electricity demand influenced by temperature (U-shaped relationship)
temp_effect <- 2000 + 100 * (temperature - 20)^2
demand <- temp_effect + 500 * sin(2 * pi * time / 52) + 
  50 * time / 52 + rnorm(n, sd = 150) + (10*time)

# Organize data as time series and separate into training and test sets
data <- tibble(
  date = seq(as.Date("2014-01-01"), by = "week", length.out = n),
  temperature = temperature,
  demand = demand
)
```

Let's examine our data. First, let's look at the time series patterns:


``` r
ggplot(data, aes(x = date)) +
  geom_line(aes(y = temperature, color = "Temperature")) +
  geom_line(aes(y = scale(demand)[,1] * sd(temperature) + mean(temperature), 
                color = "Scaled Demand")) +
  scale_color_manual(values = c("Temperature" = "#D95F5F", "Scaled Demand" = "#5A8D9B")) +
  labs(title = "Temperature and Scaled Electricity Demand Over Time", x = "Date",
       y = "Temperature (°C)", color = "Variable") +
  theme_tufte(base_size = 12) +
  theme(legend.position = "bottom")
```

![center](/figures/tsbasics/unnamed-chunk-2-1.png)

## The `modeltime` workflow for time series forecasting

We will quickly show how to use the [`modeltime`](https://business-science.github.io/modeltime/index.html) package for time series forecasts. We start by converting the data to a `tsibble`, an R object for time series dataframes. 


``` r
data_ts <- data |> 
  mutate(date = as_date(date)) |> 
  as_tsibble(index = date) |> 
  rename(demand = demand, temperature = temperature)

data_split <- initial_time_split(data, prop = holdout_prop)
train_set <- training(data_split)
holdout <- testing(data_split)
cal_set <- initial_time_split(holdout, prop = holdout_prop) |> training()
test_set <- initial_time_split(holdout, prop = holdout_prop) |> testing()
```

In order to check how well we are doing _ex-ante_, we will need to forecast the temperature into the future, to understand demand. 


``` r
temp_model_prophet <- prophet_reg() |> 
  set_engine(engine = "prophet", yearly.seasonality=TRUE) |> 
  fit(temperature ~ date, data = train_set)

temp_model_tbl <- modeltime_table(temp_model_prophet)

temp_forecast_tbl <- temp_model_tbl |> 
  modeltime_forecast(new_data = holdout, actual_data = data)

temp_forecast_tbl |> 
  plot_modeltime_forecast(.legend_show = FALSE, .title = "Temperature Forecast", .interactive = FALSE)
```

![center](/figures/tsbasics/unnamed-chunk-4-1.png)

Now, we will need to replace the temperatures in the holdout, calibration and test sets with our temperature forecasts to ensure that our demand forecasts are ex-ante.


``` r
holdout$temperature <- {temp_forecast_tbl |> filter(.key == "prediction")}$.value
cal_set <- initial_time_split(holdout, prop = holdout_prop) |> training()
test_set <- initial_time_split(holdout, prop = holdout_prop) |> testing()
data_exante <- bind_rows(train_set, holdout)
```


Now, we create a prophet model with temperature as an exogenous variable and calibrate it on the calibration set data.


``` r
model_prophet <- prophet_reg() |> 
    set_engine(engine = "prophet", yearly.seasonality=TRUE) |> 
    fit(demand ~ date+temperature, data = train_set)

model_tbl <- modeltime_table(model_prophet)

calibration_tbl <- model_tbl |> 
  modeltime_calibrate(new_data = cal_set, quiet = FALSE)
```

Now, lets see what the forecast looks like with conformal intervals calculated by `modeltime`'s built in conformal inference scheme.


``` r
forecast_tbl <- calibration_tbl |> 
  modeltime_forecast(new_data = test_set, 
                     actual_data = data_exante,
                     conf_method  = "conformal_default", 
                     conf_interval = level_conf, 
                     keep_data  = TRUE)
forecast_tbl |> 
  tail(nrow(test_set)*4) |> 
  plot_modeltime_forecast(.legend_show = TRUE, .title = "Demand Forecast", .interactive = FALSE)
```

![center](/figures/tsbasics/unnamed-chunk-7-1.png)

We see that the forecast remains equally confident far into the future which seems wrong (the regularity of the data notwithstanding), and the confidence intervals are symmetric around the mean

The modeltime workflow does calibrate the confidence intervals on samples outside the training set, so the coverage is not terrible. 

In the following sections, we will address some of these aspects using conformal methods for multi timestep prediction using the [`conformalForecast` package](https://github.com/xqnwang/conformalForecast), and describe the **AcMCP** method, introduced in [Wang and Hyndman’s paper](https://github.com/xqnwang/cpts).


### Split Conformal Inference in Time Series

In traditional split conformal inference, we:  

- Split the data into a **Training Set** for model fitting and a **Calibration Set** to compute prediction intervals.  
- For each point in the calibration set, we calculate a **nonconformity score** (typically the absolute forecast error) to quantify the forecast's deviation from the observed values.  
- By taking a quantile of these scores, we construct prediction intervals around the forecasts.
   
What we did above with modeltime calibrating on the test set was exactly 

However, **time series data introduces unique challenges**:  

- Forecast uncertainty increases with the forecast horizon due to accumulated prediction errors.
- To adapt conformal prediction for time series, we use the **sequential split** approach, which respects the order of data points and maintains the temporal structure in the data.

#### Recipe for Split Conformal Prediction

1. **Select Model and Forecast Horizon**: Choose a forecasting model (e.g., ARIMA, Prophet) and decide on the forecast horizon $H$.
2. **Split Data Sequentially**: Divide the data sequentially into a **training Set**: for fitting the model and a **calibration Set** to calculate nonconformity scores.
3. **Fit Model on Training Set**: Train the model on the training set, then use it to generate predictions for points in the calibration set.
4. **Calculate Nonconformity Scores**: For each calibration point $y_{t+h}$, calculate the forecast $\hat{y}_{t+h|t}$ and the nonconformity score:
$$s_{t+h|t} = |y_{t+h} - \hat{y}_{t+h|t}|$$

5. **Construct Prediction Intervals**: First select a confidence level $1 - \alpha$. Then, find the $(1 - \alpha)$ th quantile of nonconformity scores to construct intervals for future points:
    $$
     \hat{C}_{t+h|t} = [\hat{y}_{t+h|t} - Q_{1-\alpha}, \hat{y}_{t+h|t} + Q_{1-\alpha}]
    $$

In time series forecasting, prediction intervals ideally **widen for forecasts further into the future** to capture the increased uncertainty over time. Standard split conformal prediction doesn’t account for this, as it uses a single quantile of nonconformity scores for all horizons. However, various methods address this limitation by widening intervals adaptively:

### Horizon-Dependent Quantiles

Using horizon-specific nonconformity quantiles, we can tailor intervals to be wider for longer forecast horizons.

**Recipe**:

- For each forecast horizon $h$, calculate separate quantiles $Q_{1-\alpha}^{(h)}$ based on the nonconformity scores for that specific horizon.
- Define prediction intervals that adapt to each horizon:
   $$
    \hat{C}_{t+h|t} = [\hat{y}_{t+h|t} - Q_{1-\alpha}^{(h)}, \hat{y}_{t+h|t} + Q_{1-\alpha}^{(h)}]
   $$
- This approach allows intervals to widen as the forecast horizon increases, reflecting higher uncertainty.

## Online Conformal Inference for Multi-step Time Series Predictions

Creating prediction intervals that account for the uncertainty in predictions is challenging in time series because data points are correlated across time, violating the data exchangeability assumptions of traditional conformal methods. In particular, for multi-step forecasting (predicting $t+1, t+2, \ldots, t+H$), each interval needs to reflect both:  

1. The growing uncertainty of further predictions.
2. The dependencies between errors at different steps ahead.

In **online conformal inference** for multi-step predictions, intervals are recursively updated as new observations arrive. This lets intervals adapt to shifts in data patterns, improving the long-term accuracy of coverage over time.

## The AcMCP (Autocorrelated Multi-step Conformal Prediction) Method

The **AcMCP method** by [Wang and Hyndman (2024)](https://github.com/xqnwang/cpts/tree/main/paper) builds on these ideas by incorporating **autocorrelations** in multi-step forecast errors, unlike simpler methods that treat each forecast step independently. AcMCP produces statistically efficient prediction intervals by integrating these dependencies into the interval calculations, which is crucial for time series data.

**Recipe**  

1. **Modeling Forecast Error Structure**:  
    
    - For an $h$-step-ahead prediction, AcMCP assumes that errors follow an **MA($h-1$)** structure, meaning the current error depends on the past $h-1$ steps.
    - This structure is captured as:
    $$
     e_{t+h|t} = \omega_{t+h} + \theta_1 \omega_{t+h-1} + \cdots + \theta_{h-1} \omega_{t+1},
    $$
     where $\omega_t$ is random noise and $\theta$ terms reflect the error dependencies.

2. **Updating Quantile Estimates**: AcMCP updates the interval’s quantile estimate $q_{t+h|t}$ in real time, accounting for recent errors and their correlations. This update is key to keeping the interval valid over multiple steps and adapting to new information.

3. **Combining Multiple Models**    
    
    - An **MA($h-1$) model** trained on recent $h$-step-ahead errors to capture the correlation in errors.  
    - A **linear regression model** that uses recent errors to forecast future errors.

This combination enables AcMCP to capture both immediate and multi-step dependencies, refining prediction intervals as each new observation arrives.

Now, we illustrate the use of AcMCP using the `conformalForecast` package that accompanies the Wang and Hyndman (2024) paper. 
 








