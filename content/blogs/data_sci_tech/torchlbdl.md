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
```

```
## Error in (function (size, options) : Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```r
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

```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

Now, we can see how good our predictions are:

```r
# Get model predictions
fin_x <- seq(-1,1,length.out=100)
final_y <- model_y(fin_x)
```

```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```r
# Plotting
predicted_data <- data.frame(x = fin_x, y = as.numeric(final_y))
```

```
## Error in eval(expr, envir, enclos): object 'final_y' not found
```

```r
ggplot() +
  geom_point(data = data, aes(x, y)) +
  geom_line(data = predicted_data, aes(x, y)) +
  theme(element_text(size = 30)) +
  ggtitle("Original and Fitted Semi-Circle, 50 points and 10 weights, basis functions") +
  theme_tufte()
```

```
## Error in eval(expr, envir, enclos): object 'predicted_data' not found
```

**Underfitting** 
When the model is too small to capture the features of the data (in our case, too few weights)


```
## Error in (function (size, options) : Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```
## Error in eval(expr, envir, enclos): object 'final_y' not found
```

```
## Error in eval(expr, envir, enclos): object 'predicted_data' not found
```

**Overfitting** 
When there is too little data to properly constrain the parameters of a larger model.


```
## Error in (function (size, options) : Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```
## Error in eval(expr, envir, enclos): object 'final_y' not found
```

```
## Error in eval(expr, envir, enclos): object 'predicted_data' not found
```

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


```r
net <- nn_module(
  "ClassifierNet",
  initialize = function() {
    self$layer1 <- nn_linear(2, 2)
    self$layer2 <- nn_linear(2, 2)
    self$layer3 <- nn_linear(2, 2)
    self$layer4 <- nn_linear(2, 2)
    self$layer5 <- nn_linear(2, 2)
    self$layer6 <- nn_linear(2, 2)
    self$layer7 <- nn_linear(2, 2)
    self$output <- nn_softmax(2)
    # Initialize an environment to store activations
    self$activations <- list()
  },
  
  forward = function(x) {
    x <- self$layer1(x) |> tanh()
    self$activations$layer1 <- x
    x <- self$layer2(x) |> tanh()
    self$activations$layer2 <- x
    x <- self$layer3(x) |> tanh()
    self$activations$layer3 <- x
    x <- self$layer4(x) |> tanh()
    self$activations$layer4 <- x
    x <- self$layer5(x) |> tanh()
    self$activations$layer5 <- x
    x <- self$layer6(x) |> tanh()
    self$activations$layer6 <- x
    x <- self$layer7(x) |> tanh()
    self$activations$layer7 <- x
    x <- self$output(x)
    x
  }
)


# Convert the features and labels into tensors
features <- torch_tensor(as.matrix(data[c("x", "y")]))
```

```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```r
labels <- torch_tensor(as.integer(data$label))
```

```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```r
# Create a dataset using lists of features and labels
data_classif <- tensor_dataset(features, labels)
```

```
## Error in eval(expr, envir, enclos): object 'features' not found
```

```r
# Create a dataloader from the dataset
dataloaders <- dataloader(data_classif, batch_size = 100, shuffle = TRUE)
```

```
## Error in eval(expr, envir, enclos): object 'data_classif' not found
```

```r
# defining the model
model <- net |>
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam
  ) |>
  fit(dataloaders, epochs = 200)
```

```
## Error in cpp_backends_mps_is_available(): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

Now, let us see (visually) how well the model predicts some new synthetic data generated similarly. 


```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```
## Error in eval(expr, envir, enclos): object 'test_features' not found
```

```
## Error in eval(expr, envir, enclos): object 'model' not found
```

```
## Error in eval(ei, envir): object 'preds' not found
```

```
## Error in eval(expr, envir, enclos): object 'preds' not found
```

```
## Error in eval(expr, envir, enclos): object 'predicted_labels' not found
```

```
## Error in `geom_point()`:
## ! Problem while computing aesthetics.
## ℹ Error occurred in the 1st layer.
## Caused by error:
## ! object 'predictions' not found
```

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
```

```
## Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```r
n_class <- penguins$species |> unique() |> length() |> as.numeric()
```

Now, we train a simple MLP on 75% of this dataset. 


```r
mlpnet <- nn_module(
  "MLPnet",
  initialize = function(din, dhidden1, dhidden2, dhidden3, n_class) {
    self$net <- nn_sequential(
      nn_linear(din, dhidden1),
      nn_relu(),
      nn_linear(dhidden1, dhidden2),
      nn_relu(),
      nn_linear(dhidden2, dhidden3),
      nn_relu(),
      nn_linear(dhidden3, n_class)
    )
  },
  forward = function(x) {
    self$net(x)
  }
)

total_size <- length(ds)
```

```
## Error in eval(expr, envir, enclos): object 'ds' not found
```

```r
train_size <- floor(0.8 * total_size)
```

```
## Error in eval(expr, envir, enclos): object 'total_size' not found
```

```r
valid_size <- total_size - train_size
```

```
## Error in eval(expr, envir, enclos): object 'total_size' not found
```

```r
# Generate indices and shuffle them
set.seed(123)  # For reproducibility
indices <- sample(total_size)
```

```
## Error in eval(expr, envir, enclos): object 'total_size' not found
```

```r
train_indices <- indices[1:train_size]
```

```
## Error in eval(expr, envir, enclos): object 'indices' not found
```

```r
valid_indices <- indices[(train_size + 1):total_size]
```

```
## Error in eval(expr, envir, enclos): object 'indices' not found
```

```r
train_dataset <- ds[train_indices]
```

```
## Error in eval(expr, envir, enclos): object 'ds' not found
```

```r
valid_dataset <- ds[valid_indices]
```

```
## Error in eval(expr, envir, enclos): object 'ds' not found
```

```r
fitted_mlp <- mlpnet |> 
  setup(loss = nn_cross_entropy_loss(), optimizer = optim_adam) |> 
  set_hparams(din = 4,
              dhidden1 = 2,
              dhidden2 = 2,
              dhidden3 = 2,
              n_class = n_class) |> 
  fit(train_dataset, epochs = 15, valid_data = valid_dataset)
```

```
## Error in cpp_backends_mps_is_available(): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

Now, let us visualize the validation loss during the training process.


```
## Error in eval(expr, envir, enclos): object 'fitted_mlp' not found
```

```
## Error in eval(ei, envir): object 'train_loss' not found
```

```
## Error in eval(expr, envir, enclos): object 'fitted_mlp' not found
```

```
## Error in eval(ei, envir): object 'valid_loss' not found
```

```
## Error in eval(expr, envir, enclos): object 'train_loss' not found
```

```
## Error in eval(expr, envir, enclos): object 'loss_df' not found
```

#### Convolutional networks - resnets

Images are usually dealt with by convolutional networks - they reduce the signal size until fully connected layers can handle it, or they output 2D signals which are themselves large. Residual networks involve an architecture where signal is taken from one layer and added to a later layer. 

We will build a very simple resnet for the task of image classification, example loosely based on the DL+SC with R torch [book](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/image_classification_1.html).


```r
library(torchvision)

set.seed(777)
torch_manual_seed(777)
```

```
## Error in cpp_torch_manual_seed(as.character(seed)): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```r
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

```
## Error in cpp_backends_mps_is_available(): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

Let's see how it does on the test set. 


```r
model_eval <- evaluate(fitted, valid_dl)
```

```
## Error in cpp_backends_mps_is_available(): Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
```

```r
print(model_eval)
```

```
## Error in eval(expr, envir, enclos): object 'model_eval' not found
```

#### Attention and transformers

It seems to be a rule that any text on the internet mentioning these words must have this graphic from the original "Attention is all you need" [paper](https://arxiv.org/abs/1706.03762). 
![](attention.png).
Instead of the paper, this part of the post is largely based on [this](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) excellent article by [Sebastian Raschka](https://sebastianraschka.com/), except we will use modules built into the `torch` package instead of coding attention and transformers from scratch. 

