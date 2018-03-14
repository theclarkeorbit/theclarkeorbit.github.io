---
title: "greta playground"
output:
  html_document:
    df_print: paged
---

This notebook contains one simple worked out example with the new - still in its infancy - PPL in R with a Tensorflow backend : [Greta]( https://greta-dev.github.io/greta/index.html)

For a more mature PPL in Python also with a tensorflow backend, see [Edward](http://edwardlib.org/).



## Basic linear regression. 

We will generate artificial data with known parameters, so that we can check if Greta gets it right later. 

### Generating fake data to fit a model to


```r
length_of_data <- 100
sd_eps <- pi^exp(1)
intercept <- -5.0
slope <- pi
x <- seq(-10*pi, 10*pi, length.out = length_of_data)
y <- intercept + slope*x + rnorm(n = length_of_data, mean = 0, sd = sd_eps)
data <- data_frame(y = y, x = x)
ggplot(data, aes(x = x, y = y)) +
  geom_point() +
  geom_smooth(method = 'lm') +
  ggtitle("Fake experimental data")
```

![center](/figures/greta_playground/unnamed-chunk-2-1.png)

Given this data, we want to write Greta code to infer the model parameters. Since we really don't know anything about the prior distributions of our parameters, we look at the experimental data and take rough, uniform priors. 

### Defining clueless priors for model parameters


```r
intercept_p <- uniform(-10, 10)
sd_eps_p <- uniform(0, 50)
slope_p <- uniform(0, 10)
```

Now, we define and plot the model.

### Defining the model


```r
mean_y <- intercept_p+slope_p*x
distribution(y) <- normal(mean_y, sd_eps_p)

our_model <- model(intercept_p, slope_p, sd_eps_p)
plot(our_model)
```

### Fitting model to data

Now, we sample from our model

```r
num_samples <- 1000
param_draws <- mcmc(our_model, n_samples = num_samples, warmup = num_samples / 10)
```

and plot the samples, and the parameter fits.

```r
mcmc_dens(param_draws)
```

![center](/figures/greta_playground/unnamed-chunk-6-1.png)

```r
mcmc_intervals(param_draws)
```

![center](/figures/greta_playground/unnamed-chunk-6-2.png)

By inspection, it looks like the [HMC](https://arxiv.org/abs/1701.02434) has reached some reasonable values for our model parameters. 

Explicitly, the mean estimates can be computed from the `param_draws` data structure, or via the `greta::opt` function.   

```r
param_draws_df <- as_data_frame(param_draws[[1]])
param_estimates <- param_draws_df %>% 
  summarise_all(mean)
param_estimates %>% print()
```

```
## # A tibble: 1 x 3
##   intercept_p slope_p sd_eps_p
##         <dbl>   <dbl>    <dbl>
## 1       -5.95    3.02     21.2
```

```r
opt_params <- opt(our_model)
opt_params$par %>% print()
```

```
## intercept_p     slope_p    sd_eps_p 
##   -5.106374    2.980471   19.102325
```

### Plotting against data

the `calculate()` function is available on the latest release of `greta` on github.

```r
mean_y_plot <- intercept_p+slope_p*x
mean_y_plot_draws <- calculate(mean_y_plot, param_draws)
mean_y_est <- colMeans(mean_y_plot_draws[[1]])
data_pred <- data %>% mutate(y_fit = mean_y_est)
ggplot(data_pred) +
    geom_point(aes(x,y), colour = "blue") +
    geom_smooth(aes(x,y), colour = 'blue', method = 'lm') +
    geom_line(aes(x,y_fit), colour = 'red')
```

![center](/figures/greta_playground/unnamed-chunk-8-1.png)

