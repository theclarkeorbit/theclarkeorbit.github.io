---
title: "Basics of conformal prediction in time series data"
Date: 01 Aug 2024
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
4. [Frank Hull's](https://x.com/frankiethull) R code for [Valeriy Manokhin's](https://x.com/predict_addict) [book](https://www.amazon.in/dp/1805122762?ref_=cm_sw_r_cp_ud_dp_H765W8M42WJ6XJVNZGD7_1). 


Conformal prediction is a powerful, non-parametric approach to constructing prediction intervals that provide reliable coverage without requiring strong assumptions about the data distribution (see the [previous post](https://theclarkeorbit.github.io/tidymodels-and-conformal-prediction.html) for an introduction and reading material). For time series forecasting, where data is temporally correlated, conformal prediction methods must adapt to these correlations to achieve accurate and reliable intervals.

We will first generate time series data and illustrate a couple of classical forecasting techniques before moving to conformal methods.

### Generating simple time series data

Let's create a synthetic dataset that mimics real-world electricity demand and temperature patterns, where demand is influenced by temperature in a U-shaped relationship (higher demand at both very low and very high temperatures):


``` r
set.seed(123)
n <- 52*5  # 5 years of weekly data
time <- 1:n

# Generate synthetic temperature data with seasonal pattern
temperature <- 20 + 10 * sin(2 * pi * time / 52) + rnorm(n, sd = 2)

# Generate electricity demand influenced by temperature (U-shaped relationship)
temp_effect <- 2000 + 100 * (temperature - 20)^2
demand <- temp_effect + 500 * sin(2 * pi * time / 52) + 
  50 * time / 52 + rnorm(n, sd = 100) + (10*time)

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
  labs(title = "Temperature and Scaled Electricity Demand Over Time",
       y = "Temperature (°C)", color = "Variable") +
  theme_tufte(base_size = 12) +
  theme(legend.position = "top")
```

![center](/figures/tsbasics/unnamed-chunk-2-1.png)

## The `modeltime` workflow for time series forecasting

We will quickly show how to use the [`modeltime`](https://business-science.github.io/modeltime/index.html) package for time series forecasts. We start by converting the data to a `tsibble`, an R object for time series dataframes. 


``` r
data_ts <- data |> 
  mutate(date = as_date(date)) |> 
  as_tsibble(index = date) |> 
  rename(demand = demand, temperature = temperature)

data_split <- initial_time_split(data, prop = 0.75)
train_set <- training(data_split)
holdout <- testing(data_split)
cal_set <- initial_time_split(holdout, prop = 0.8) |> training()
test_set <- initial_time_split(holdout, prop = 0.8) |> testing()
```

In order to check how well we are doing _ex ante_, we will need to forecast the temperature into the future, to understand demand. 


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

Now, we will need to replace the temperatures in the holdout, calibration and test sets with our temperature forecasts to ensure that are demandforecasts are ex ante.


``` r
holdout$temperature <- {temp_forecast_tbl |> filter(.key == "prediction")}$.value
cal_set <- initial_time_split(holdout, prop = 0.8) |> training()
test_set <- initial_time_split(holdout, prop = 0.8) |> testing()
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
                     actual_data = data,
                     conf_method  = "conformal_default", 
                     conf_interval = 0.80, 
                     keep_data  = TRUE)
forecast_tbl |> 
  plot_modeltime_forecast(.legend_show = FALSE, .title = "Demand Forecast", .interactive = FALSE)
```

![center](/figures/tsbasics/unnamed-chunk-7-1.png)

We see that the forecast remains equally confident far into the future which seems wrong (the regularity of the data notwithstanding), and the confidence intervals are symmetric around the mean, not terribly useful. 

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

1. **Select Model and Forecast Horizon**: Choose a forecasting model (e.g., ARIMA, Prophet) and decide on the forecast horizon$H$.
2. **Split Data Sequentially**:  
- Divide the data sequentially into:  
  - **Training Set**: for fitting the model.
  - **Calibration Set**: to calculate nonconformity scores.
3. **Fit Model on Training Set**: Train the model on the training set, then use it to generate predictions for points in the calibration set.
4. **Calculate Nonconformity Scores**: For each calibration point$y_{t+h}$, calculate the forecast$\hat{y}_{t+h|t}$and the nonconformity score:
$$s_{t+h|t} = |y_{t+h} - \hat{y}_{t+h|t}|$$

5. **Construct Prediction Intervals**:

- Select a confidence level$1 - \alpha$.
- Find the$(1 - \alpha)$-quantile of nonconformity scores to construct intervals for future points:
    $$
     \hat{C}_{t+h|t} = [\hat{y}_{t+h|t} - Q_{1-\alpha}, \hat{y}_{t+h|t} + Q_{1-\alpha}]
    $$

In time series forecasting, prediction intervals ideally **widen for forecasts further into the future** to capture the increased uncertainty over time. Standard split conformal prediction doesn’t account for this, as it uses a single quantile of nonconformity scores for all horizons. However, various methods address this limitation by widening intervals adaptively:

### Horizon-Dependent Quantiles

Using horizon-specific nonconformity quantiles, we can tailor intervals to be wider for longer forecast horizons.

**Recipe**:

- For each forecast horizon$h$, calculate separate quantiles$Q_{1-\alpha}^{(h)}$based on the nonconformity scores for that specific horizon.
- Define prediction intervals that adapt to each horizon:
   $$
    \hat{C}_{t+h|t} = [\hat{y}_{t+h|t} - Q_{1-\alpha}^{(h)}, \hat{y}_{t+h|t} + Q_{1-\alpha}^{(h)}]
   $$
- This approach allows intervals to widen as the forecast horizon increases, reflecting higher uncertainty.

## Online Conformal Inference for Multi-step Time Series Predictions

Creating prediction intervals that account for the uncertainty in predictions is challenging in time series because data points are correlated across time, violating the data exchangeability assumptions of traditional conformal methods. In particular, for multi-step forecasting (predicting$t+1, t+2, \ldots, t+H$), each interval needs to reflect both:  

1. The growing uncertainty of further predictions.
2. The dependencies between errors at different steps ahead.

In **online conformal inference** for multi-step predictions, intervals are recursively updated as new observations arrive. This lets intervals adapt to shifts in data patterns, improving the long-term accuracy of coverage over time.

## The AcMCP (Autocorrelated Multi-step Conformal Prediction) Method

The **AcMCP method** by [Wang and Hyndman (2024)](https://github.com/xqnwang/cpts/tree/main/paper) builds on these ideas by incorporating **autocorrelations** in multi-step forecast errors, unlike simpler methods that treat each forecast step independently. AcMCP produces statistically efficient prediction intervals by integrating these dependencies into the interval calculations, which is crucial for time series data.

**Recipe**  

1. **Modeling Forecast Error Structure**:

- For an$h$-step-ahead prediction, AcMCP assumes that errors follow an **MA($h-1$)** structure, meaning the current error depends on the past$h-1$steps.
- This structure is captured as:
    $$
     e_{t+h|t} = \omega_{t+h} + \theta_1 \omega_{t+h-1} + \cdots + \theta_{h-1} \omega_{t+1},
    $$
     where$\omega_t$is random noise and$\theta$terms reflect the error dependencies.

2. **Updating Quantile Estimates**: AcMCP updates the interval’s quantile estimate$q_{t+h|t}$in real time, accounting for recent errors and their correlations. This update is key to keeping the interval valid over multiple steps and adapting to new information.

3. **Combining Multiple Models**  

- An **MA($h-1$) model** trained on recent$h$-step-ahead errors to capture the correlation in errors.  
- A **linear regression model** that uses recent errors to forecast future errors.

   This combination enables AcMCP to capture both immediate and multi-step dependencies, refining prediction intervals as each new observation arrives.

AcMCP has some advantages that offset the extra computation needed to include the autocorrelations in the errors while generating the conformal predictons:  

- **Efficiency**: AcMCP’s intervals are generally tighter and more accurate, thanks to the autocorrelation adjustments.  
- **Robustness for Long-range Forecasts**: AcMCP is ideal for long-horizon forecasts, as it considers error dependencies over multiple steps.  
- **Dynamic Adjustments**: By updating intervals with each new data point, AcMCP provides accurate coverage as time progresses.

This method provides a robust way to build multi-step prediction intervals that are both statistically efficient and resilient to changes in the forecast horizon.

Now, we illustrate the use of AcMCP using the `conformalForecast` package that accompanies the Wang and Hyndman (2024) paper. 
 

``` r
horizon <- nrow(test_set)
# function that makes predictons of demand
demand_forecast <- function(y, h, level, xreg, newxreg) {
    model <- auto.arima(y, xreg = xreg)
    # Forecast using future temperature values
    fc <- forecast(model, h = h, xreg = newxreg, level = level)
    return(fc)  # Returns a forecast object
}

# Generateing rolling forecast and errors on the given data
demand_fc <- cvforecast(
    y = ts({rbind(train_set, cal_set) |> select(demand)}, frequency = 52),
    forecastfun = demand_forecast,
    h = horizon,
    level = c(80),
    xreg = matrix({data$temperature |> head(nrow(train_set)+nrow(cal_set)+horizon)}, ncol = 1),
    initial = 10, 
    window = horizon*10
)
```

We use the forecasting function created above to generate conformal predictions.


``` r
cal_window <- max(10, horizon*5)
symm <- FALSE
roll <- TRUE
Tg <- 15
delta <- 0.01
Csat <- 2 / pi * (ceiling(log(Tg) * delta) - 1 / log(Tg))
KI <- 0.5
lr <- 0.1

acmcp <- mcp(demand_fc, alpha = 1 - 0.01 * demand_fc$level,
             ncal = cal_window, rolling = roll,
             integrate = TRUE, scorecast = TRUE,
             lr = lr, KI = KI, Csat = Csat)
```


Lets see how this compares with the forecasts from the modeltime prophet engine. First we need to extract the forecast from the `acmcp` object.


``` r
acmcp_df <- tibble(
  date = {head(data$date, {nrow(train_set) + nrow(cal_set) + horizon}) |> tail(horizon)}, 
  demand = {head(data$demand, {nrow(train_set) + horizon}) |> tail(horizon)},
  acmcp_lower = acmcp$lower,
  acmcp_upper = acmcp$upper,
  acmcp_forecast = acmcp$mean
)
```

Let's now compare it to the `modeltime` conformal prediction,


``` r
forecast_comparison_df <- forecast_tbl |>
  filter(.key %in% c("prediction")) |>
  select(date, .value, .conf_lo, .conf_hi) |>
  rename(
    modeltime_forecast = .value,
    modeltime_lower = .conf_lo,
    modeltime_upper = .conf_hi
  ) |>
  right_join(acmcp_df, by = "date")  # Join with AcMCP results

bind_rows(train_set, cal_set, forecast_comparison_df) |> tail(horizon*3) -> forecast_comparison_df

ggplot(forecast_comparison_df, aes(x = date)) +
  geom_line(aes(y = demand, color = "Actual Demand"), size = 0.7) +
  geom_line(aes(y = modeltime_forecast, color = "Modeltime Forecast"), linetype = "dotted", size = 0.7) +
  geom_ribbon(aes(ymin = modeltime_lower, ymax = modeltime_upper, fill = "Modeltime Conformal Interval"), alpha = 0.2) +
  geom_line(aes(y = acmcp_forecast, color = "AcMCP Forecast"), linetype = "dashed", size = 0.9) +
  geom_ribbon(aes(ymin = acmcp_lower, ymax = acmcp_upper, fill = "AcMCP Conformal Interval"), alpha = 0.3) +
  scale_color_manual(values = c("Actual Demand" = "black", "Modeltime Forecast" = "#5A8D9B", "AcMCP Forecast" = "#D95F5F")) +
  scale_fill_manual(values = c("Modeltime Conformal Interval" = "#5A8D9B", "AcMCP Conformal Interval" = "#D95F5F")) +
  labs(
    title = "Comparison of Conformal Prediction Intervals: Modeltime vs AcMCP",
    y = "Demand",
    x = "Date"
    fill = "Confidence Interval",
    color = "Forecast Type"
  ) +
  theme_tufte(base_size = 12) +
  theme(legend.position = "bottom") +
  guides(
    color = guide_legend(nrow = 3),
    fill = guide_legend(nrow = 3)
  )
## Error: <text>:25:5: unexpected symbol
## 24:     x = "Date"
## 25:     fill
##         ^
```

AcMCP, for the cost of complexity and computation, does give us confidence intervals that are more adaptive and ensure coverage into the uncertain future. However, it is not clear (to me, yet) how one could incorporate this into a production workflow to make actual online conformal predictions and keep track of them over time.
