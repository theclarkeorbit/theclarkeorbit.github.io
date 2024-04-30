---
title: "R, Torch, and the little book of deep learning"
Date: 29 Apr 2024
output:
  html_document:
    df_print: paged
---

These notes are meant to implement little examples from [Francois Fleuret's](https://fleuret.org/francois/index.html) Little Book of Deep Learning ([pdf link](https://fleuret.org/public/lbdl.pdf)) in r-torch. I'm writing these as a fun way to dive into torch in R while surveying DL quickly. 

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
  theme(element_text(size = 30)) +
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
# Learning rate
lr <- 0.01

# Gradient descent loop
for (epoch in 1:5000) {
  y_pred <- model_y(x)
  
  op <- nnf_mse_loss(y_pred, torch_tensor(y))
  
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
  theme(element_text(size = 30)) +
  ggtitle("Original and Fitted Semi-Circle, 50 points and 10 weights, basis functions") +
  theme_tufte()
```

![center](/figures/torchlbdl/unnamed-chunk-5-1.png)

**Underfitting** 
When the model is too small to capture the features of the data (in our case, too few weights)

![center](/figures/torchlbdl/unnamed-chunk-6-1.png)

**Overfitting** 
When there is too little data to properly constrain the parameters of a larger model.

![center](/figures/torchlbdl/unnamed-chunk-7-1.png)

### Classification, and the usefulness of depth

Let us generate data that looks similar to that shown in section 3.5 of the book. 


```r
# Function to generate C-shaped data
generate_c_data <- function(n, x_offset, y_offset, y_scale, label, xflipaxis) {
  theta <- runif(n, pi/2, 3*pi/2)  # Angle for C shape
  x <- cos(theta) + rnorm(n, mean = 0, sd = 0.1) + x_offset
  if(xflipaxis==T){
    x <- 1-x
  }
  y <- sin(theta) * y_scale + rnorm(n, mean = 0, sd = 0.1) + y_offset
  data.frame(x = x, y = y, label = label)
}

# Number of points per class
n_points <- 1000

# Generate data for both classes
data_class_1 <- generate_c_data(n_points, x_offset = 0, 
                                y_offset = 0, y_scale = 1, label = 1,
                                xflipaxis = F)
data_class_0 <- generate_c_data(n_points, x_offset = 1.25, 
                                y_offset = -1.0, y_scale = -1, label = 2,
                                xflipaxis = T)  # Mirrored and adjusted

# Combine data
data <- rbind(data_class_0, data_class_1)

# Plotting the data
ggplot(data, aes(x, y, color = as.factor(label))) +
  geom_point(alpha = 0.7, size = 3) +
  theme(element_text(size = 30)) +
  labs(title = "Adjusted C-shaped Data for Classification", x = "X axis", y = "Y axis") +
  theme_tufte()
```

![center](/figures/torchlbdl/unnamed-chunk-8-1.png)

Now, we can build a very simple neural net to classify these points and try to visualize what the trained net is doing at each layer.



Now, let us see (visually) how well the model predicts some new synthetic data generated similarly. 

![center](/figures/torchlbdl/unnamed-chunk-10-1.png)

**TODO:** I have not yet figured out a way to access model activations when the model is trained with luz.


### Architectures

#### Multi Layer Perceptrons

This is a neural net that has a series of fully connected layers seperated by activations. We will illustrate an MLP using the Penguins dataset, where we try to predict the species of a penguin from some features. Example adapted from the excellent Deep Learning with R Torch [book](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/). 


```r
library(palmerpenguins)

penguins <- na.omit(penguins)
ds <- tensor_dataset(
  torch_tensor(as.matrix(penguins[, 3:6])),
  torch_tensor(
    as.integer(penguins$species)
  )$to(torch_long())
)

n_class <- penguins$species |> unique() |> length() |> as.numeric()
```

Now, we train a simple MLP on 75% of this dataset. 



Now, let us visualize the validation loss during the training process.

![center](/figures/torchlbdl/unnamed-chunk-13-1.png)

#### Convolutional networks - resnets

Images are usually dealt with by convolutional networks - they reduce the signal size until fully connected layers can handle it, or they output 2D signals which are themselves large. Residual networks involve an architecture where signal is taken from one layer and added to a later layer. 

We will build a very simple resnet for the task of image classification, example loosely based on the DL+SC with R torch [book](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/image_classification_1.html).


```r
library(torchvision)

set.seed(777)
torch_manual_seed(777)

dir <- "~/.torch-datasets"

train_ds <- torchvision::kmnist_dataset(train = TRUE,
  dir,
  download = TRUE,
  transform = function(x) {
    x |>
      transform_to_tensor() 
  }
)

valid_ds <- torchvision::mnist_dataset(train = FALSE,
  dir,
  transform = function(x) {
    x |>
      transform_to_tensor()
  }
)

train_dl <- dataloader(train_ds,
  batch_size = 128,
  shuffle = TRUE
)
valid_dl <- dataloader(valid_ds, batch_size = 128)
```

There, we have downloaded the Kanji MNIST dataset to use to test our simple resnet. The model below is partially based on [this tutorial](https://jtr13.github.io/cc21fall2/tutorial-on-r-torch-package.html) from 2021.


```r
# Define a simple Residual Block
simple_resblock <- nn_module(
  "SimpleResBlock",
  initialize = function(channels) {
    self$conv1 <- nn_conv2d(channels, channels, kernel_size = 3, padding = 1)
    self$relu1 <- nn_relu()
    self$conv2 <- nn_conv2d(channels, channels, kernel_size = 3, padding = 1)
    self$relu2 <- nn_relu()
  },
  forward = function(x) {
    identity <- x
    out <- self$relu1(self$conv1(x))
    out <- self$relu2(self$conv2(out))
    out + identity #the eponymous residual operation.
  }
)


net <- nn_module(
  "Net",
  
  initialize = function() {
    self$conv1 <- nn_conv2d(1, 32, 3, 1)
    self$conv2 <- nn_conv2d(32, 64, 3, 1)
    self$dropout1 <- nn_dropout(0.25)
    self$dropout2 <- nn_dropout(0.5)
    self$fc1 <- nn_linear(9216, 128)
    self$resblock1 <- simple_resblock(32) # 32 since its used after first conv layer that o/ps 32 channels
    self$resblock2 <- simple_resblock(64) # used after the second conv layer that outputs 64 channels
    self$fc2 <- nn_linear(128, 10)
  },
  
  forward = function(x) {
    x |>                                  # N * 1 * 28 * 28
      self$conv1() |>                     # N * 32 * 26 * 26
      nnf_relu() |>     
      self$resblock1() |>                 # the residual block
      self$conv2() |>                     # N * 64 * 24 * 24
      nnf_relu() |> 
      self$resblock2() |>                 # second residual block
      nnf_max_pool2d(2) |>                # N * 64 * 12 * 12
      self$dropout1() |> 
      torch_flatten(start_dim = 2) |>     # N * 9216
      self$fc1() |>                       # N * 128
      nnf_relu() |> 
      self$dropout2() |> 
      self$fc2()                           # N * 10
  }
)
```

Now, we will train this on our data. 


```r
fitted <- net |>
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) |>
  fit(train_dl, epochs = 1)
```

Let's see how it does on the test set. 


```r
model_eval <- evaluate(fitted, valid_dl)
print(model_eval)
```

```
## A `luz_module_evaluation`
## ── Results ─────────────────────────────────────────────────────────────────────
## loss: 0.3452
## acc: 0.899
```

#### Attention and transformers

[TODO]
