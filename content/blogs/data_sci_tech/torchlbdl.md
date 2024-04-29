---
title: "A little r-torch"
Date: 29 Apr 2024
output:
  html_document:
    df_print: paged
---

# Notes on the little book of deep learning, with torch

These notes are meant to impliment little examples from [Francois Fleuret's](https://fleuret.org/francois/index.html) Little Book of Deep Learning ([pdf link](https://fleuret.org/public/lbdl.pdf)) in r-torch. I'm writing these as a fun way to dive into torch in R while surveying DL quickly. 

You'll need to have the book with you to understand these notes, but since it is available freely, that ought not to be an issue.




### Basis function regression

We will generate synthetic data that is similar to the example shown in the book:


```r
# Number of samples
n <- 50

# Randomly distributed x values from -1 to 1
set.seed(123) # for reproducibility
x <- runif(n, min = -1, max = 1)
```

```
## 
## Attaching package: 'stats'
```

```
## The following objects are masked from 'package:dplyr':
## 
##     filter, lag
```

```r
x <- sort(x) # Sorting for plotting purposes

# y values as the semi-circle
y <- sqrt(1 - x^2)

# Plotting the original semi-circle with randomly distributed x
data <- data.frame(x = x, y = y)
ggplot(data, aes(x, y)) + 
  geom_point() + 
  ggtitle("Semi-Circle") +
  theme_tufte()
```

![center](/figures/torchlbdl/unnamed-chunk-1-1.png)

Following the book, we use gaussian kernels as the basis functions to fit $y \sim f(x;w)$, where $w$ are the weights of the basis functions.


```r
# Define Gaussian basis functions
basis_functions <- function(x, centers, scale = 0.1) {
  exp(-((x - centers)^2) / (2 * scale^2))
}

# Centers of the Gaussian kernels, these do cover the region of space we are interested in
centers <- seq(-1, 1, length.out = 10)
```

Now, we define our model for $y$, which, for basis function regression, is a linear combination of the basis functions initialized with random weights $w$. 


```r
# Initial random weights
weightss <- torch_randn(length(centers), requires_grad = TRUE)

# Calculate the model output
model_y <- function(x) {
  # Convert x to a torch tensor if it isn't already one
  x_tensor <- torch_tensor(x)
  
  # Create a tensor for the basis functions evaluated at each x
  # Resulting tensor will have size [length(x), length(centers)]
  basis_matrix <- torch_stack(lapply(centers, function(c) basis_functions(x_tensor, c)), dim = 2)
  
  # Calculate the output using matrix multiplication
  # basis_matrix is [n, 10] and weights is [10, 1]
  y_pred <- torch_matmul(basis_matrix, weightss)
  
  # Flatten the output to match the dimension of y
  return(y_pred)
}
```

Now,we will use gradient descent to minimise the MSE between the model and the real values of $y$ to obtain the optimal weights $w^*$. 


```r
# Define the loss function (MSE)
loss_fn <- function(y_pred, y_true) {
  (y_pred - y_true)^2 |>  mean()
}

# Learning rate
lr <- 0.01

# Gradient descent loop
for (epoch in 1:5000) {
  y_pred <- model_y(x)
  #loss <- nn_mse_loss()
  
  op <- loss_fn(y_pred, torch_tensor(y))
  
  # Backpropagation
  op$backward()
  
  # Update weights
  with_no_grad({
    weightss$sub_(lr * weightss$grad)
    weightss$grad$zero_()
  })
}
```

Now, we can see how good our predictions are:

```r
# Get model predictions
fin_x <- seq(-1,1,length.out=100)
final_y <- model_y(fin_x)

# Plotting
predicted_data <- data.frame(x = fin_x, y = as.numeric(final_y))
ggplot() +
  geom_point(data = data, aes(x, y)) +
  geom_line(data = predicted_data, aes(x, y)) +
  ggtitle("Original and Fitted Semi-Circle, 50 points and 10 weights, basis functions") +
  theme_tufte()
```

![center](/figures/torchlbdl/unnamed-chunk-5-1.png)

Underfitting: when the model is too small to capture the features of the data (in our case, too few weights)

![center](/figures/torchlbdl/unnamed-chunk-6-1.png)

Overfitting: when there is too little data to properly constrain the parameters of a larger model.

![center](/figures/torchlbdl/unnamed-chunk-7-1.png)
