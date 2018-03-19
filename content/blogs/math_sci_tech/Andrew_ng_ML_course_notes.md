---
title: "Notes from the original ML course by Andrew Ng"
date: 24 Feb 2016
output:
  html_document:
    df_print: paged
---

These are notes I took while watching the lectures from [Andrew Ng's ML course](https://www.coursera.org/learn/machine-learning). There is no code, just some math and my take aways from the course. It should still serve as a useful first document to skim for someone just starting out with machine learning.





## What can ML do ?

**Mining large data sets**  
Automation and digitization has led to huge data sets which can be mined, and used using machine learning techniques. Predictions become possible because sophisticated algorithms can learn from large data sets.

**Problems that cannot be programmed by hand**  
Flying a helicopter. It is very hard to program a computer to do this complex task, but a good neural network can learn to do it via a reasonable training program.

## Definitions of ML
    
Arthur Samuel : "The field of study that gives computers the ability to learn without being explicitly programmed."

Tom Mitchell : "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Broad types of machine learning :

1. Supervised learning   
2. Unsupervised learning   

## Supervised learning

The learning algorithm is provided with a dataset that includes the correct values for the target variable. So, in case we want to predict house prices as a function of size, then a data set that has a list of house sizes and the corresponding prices will be provided to an algorithm. Once the algorithm "leans" from this **training set** and constructs a **model** for house prices, this model can be used to **predict** the prices of houses whose sizes are known listed in some **test set**. 

When the target variable (here price) is *continuous*, the problem is known as a **regression** problem. If the target variable is *discrete* (e.g.. if we wanted to predict which zip code a house was in based on its size and price) the problem is called a **classification** problem.

## Unsupervised learning

Here, the problem is not to predict the value of a particular variable, but to detect some structure or pattern in the data set. A common example is **clustering** where the points in the data set are grouped into clusters that are somehow similar to each other. Examples :

1. Google news clusters similar news stories together.  
2. Genomics. Expression levels of thousands of genes are measured in various situations and genes which seem to be related to each other are identified using clustering.  
3. Market segmentation for marketing.  
4. Cocktail party algorithm. Separate the voices of different people at a party by identifying sounds with the same characteristics.   

and other such applications. 

## Models and cost functions

Recall that a supervised learning algorithm takes a data set (the **training set** with $m$ examples (rows)) and learns from it to construct a **model**. We denote the *features* or *predictors* by the letter $x$ while the target variable is $y$. $(x^i, y^i)$ is one row of the training set. $x^i$ is a vector with as many elements as there are features. For convenience, we will assume the number of features (predictors, columns) to be $n$. 

We denote the model by $h$ (for hypothesis) this is a function that maps from $x$ to $y$. If $h$ is parametrized by some set of parameters $\theta$, we can write $y=h_{\theta}(x)$. When the model is linear, we call this **linear regression**.   

\[
y=h_{\theta}(x)=\sum_{j}\theta_j x_j=x\theta
\]

where $x$ is a $1*n$ vector and $\theta$ is a $n*1$ vector of parameters. Note that the first row of data is always 1, so that the first parameter is always a bias value. 

We determine the values of the model parameters $\theta$ that will result in the best possible prediction of $y$ given $x$. We do this by defining a cost function (something that measures the error in predictions from our model) and minimise this cost function to obtain the final form of our model. This is an *optimization problem*. So, one might want to find $\theta$ such that $E\left[\sum_i(h_{\theta}(x^i)-y^i)^2\right]$ is minimized. So, we minimise the expectation value of the squared error.  

\[
J(\theta) = E\left[\sum_i(h_{\theta}(x^i)-y^i)^2\right]
\]

This is called the squared error cost function and is commonly used for linear regression problems. Other cost functions are possible, but generally the form of the cost function that is used is determined by how efficiently it can be minimized. One common way of writing the above cost function is 

\[J(\theta) = \frac{1}{2m}\left[\sum_{i=1}^m (h_{\theta}(x^i)-y^i)^2\right] \]

where the factor of $\frac{1}{2}$ is added by convention. Representing by $X$ the matrix of the data (excluding the target variable) where predictors are columns and each example is in a different row ($X$ is an $m*n$ matrix), and letting $y$ be the $n*1$ vector with the target variables, we can write down the matrix version of the cost function for linear regression ($h_{\theta}(x^i)=x^i\theta$, the product of the vector of parameters $\theta$ and the $i^{th}$ row of the data) as follows,

\[J(\theta) = \frac{1}{2m}(X \theta-y)^T (X \theta-y).\]

Now, the problem of learning this linear model is reduced to searching for $\theta^{*}$ in the multi dimensional space $\{\theta\}$ for which the cost function $J$ is minimised for the given training set. This is achieved (in general) using something called a **gradient descent algorithm**. 


\[
\theta^{*} = \arg \min_{\theta}\left(E\left[\sum_i(h_{\theta}(x^i)-y^i)^2\right]\right)
\]

**Notation alert :** $x^i_j$ denotes the $j^{th}$ feature in the $i^{th}$ row of the training set. 

## Gradient Descent

The basic prescription is the following :

1. Start with a random initial guess $\theta_0$.  
2. Find the direction in the space $\{\theta\}$ in which the cost function $J$ decreases the most.  
3. Take a baby step in this direction and repeat step 2, until a minimum is reached.   

We determine the "direction of maximum decrease" for $J(\theta)$ using a derivative. 

\[
\text{repeat until convergence} \left[\theta_j := \theta_j-\alpha\frac{dJ(\theta)}{d\theta_j} \text{  }\forall j\right]
\]

where $\alpha$ determines the size of our baby step (the *learning rate*), and we are just walking in the direction of the [gradient](https://en.wikipedia.org/wiki/Gradient) $-\nabla_\theta J$. The symbol $:=$ here is an assignment, not a truth assertion. It is implicit that all the $\theta$ are updated simultaneously, and that the cost function $J$ is differentiable with respect to all the $\theta$. This is what makes the mean square error a good cost function - some other possible cost functions (like the absolute error) are not differentiable.

There are various flavours of gradient descent. If all the samples of the training set are used to compute the gradient at every step (as described above) the algorithm is called *batch gradient descent*.

For linear regression, all the derivatives $\frac{dJ(\theta)}{d\theta_j}$ are trivial to compute (see [this blog post](http://eli.thegreenplace.net/2015/the-normal-equation-and-matrix-calculus/) for an excellent explanation for derivatives on matrices) and the gradient descent algorithm can be written as,

\[\text{repeat until convergence}\left[\theta_j :=\theta_j - \alpha \frac{1}{m}\sum_i (x^i\theta-y^i)x^i_j\right]\]

which, in matrix form becomes,

\[\text{repeat until convergence}\left[\theta := \theta - \frac{\alpha}{m}(X\theta-y)^TX \right].\]

### Practical tips for gradient descent

**Feature scaling :** When different features are on very different scales, the hills/valleys we would like to reach in our gradient descent optimization are shaped like long narrow canyons, and along the length, the gradient descent algorithm converges very slowly to the minimum/maximum. If we scale features so that the hills/valleys have more "circular" symmetry, gradient descent converges faster. It is better to have all features scaled into the same range of values, say $[-1,1]$ or $[0,1]$. 

A common way to scale a feature $j$ would be
\[
v_j = \frac{x_j-\mu_j}{\sigma_j}
\]

where $\mu_j$ is the mean and $\sigma_j$ is the standard deviation of the values taken by feature $j$ in the training set, and $v_j$ are the new scaled values of the feature $j$.

**Learning rate $\alpha$ :** If gradient descent is working properly, $J$ should decrease after every iteration, and convergence is defined by $J$ decreasing by some very small value at each iteration (say $10^{-4}$ or so). If gradient descent is blowing up ($J$ is *increasing* with each iteration) it could be because the learning rate $\alpha$ is too large and the algorithm is "overshooting" the optimum. Decreasing $\alpha$ should fix this. For a small enough $\alpha$ gradient descent should ALWAYS find a local optimum, and $J$ should decrease with every iteration. 

On the other hand, a very small $\alpha$ will lead to very slow convergence. 

### Polynomial regression

This is a generalization of linear regression which we saw in previous sections. now, instead of the simple linear from $h=\theta^Tx$, we also include higher powers of each factor. So, we list our new features as $u_0 = x_0 = 1, u_{1} = x^{f_1}_1,u_{2} = x^{f_2}_1\cdots u_{k} = x^{f_k}_2\cdots \forall k, \forall x$. Thus, we include in our model all powers of each factor that we think are relevant, including fractional powers. Now, the hypothesis function (model) is defined as usual, $h=\theta^Tu$ on the new feature set $\{u\}$, and the same principles off linear regression discussed earlier apply.

Thus, polynomial regression is an example of something that is ubiquitous in machine learning applications - **feature engineering**. Machine algorithms find optima faster, and predict better when certain functions of the features available in the data are also included as inputs. Finding such functions can be a matter of intuition and experience as well as thorough data exploration. [Kaggle](www.kaggle.com) contests are an example where it is clever feature engineering more than anything else that determines performance of machine learning algorithms.

## The normal equation for linear regression

For linear regression, one can solve for $\theta$ analytically, without the iterative gradient descent procedure. 
The problem is to minimize $J(\theta_0,\theta_1,\cdots\theta_n) = E_i[h_{\theta}(x^i)-y^i]$ wrt the $\theta$s. This requires the solution of $(n+1)$ equations

\[
\frac{d}{d\theta_0}J(\theta)=\frac{d}{d\theta_1}J(\theta)=\cdots=0
\]

For the particular quadratic cost function we have used, the solution is given by

\[
\theta^*=(X^TX)^{-1}X^Ty
\]
where $X$ is the matrix of all features in the training set (including $x_0=1$) and $y$ is the vector of target variables in the training set. For the derivation (express $J$ in matrix form, and differentiate wrt $\theta$ and set the derivatives to 0, there are subtleties while differentiating matrices, transposes wrt vectors) see page 45 of [ESLR](http://statweb.stanford.edu/~tibs/ElemStatLearn/). See [this blog](http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/) for a friendly explanation of the derivation sketched out in ESLR.

The normal equation is excellent when the matrix $X$ is small and the number of features is $<10000$. The matrix operations become slow for large data sets, and then gradient descent is the fall back option.

In the normal equation, we see that we need to invert the matrix $X^TX$ and many things can make a matrix non-invertible. Generally, the solution is to delete some features that might be redundant and/or [regularize](https://en.wikipedia.org/wiki/Regularization_%28mathematics%29) the matrix. Regularization prevents over-fitting when the number of variables are greater than the number of equations (in which case of course, a unique solution cannot be found).

## Logistic regression, classification

When the target variable is discrete, the problem is called a classification problem. E.g.. is an email spam/not spam ? is a tumour malignant/non malignant ? Needless to say, it is not a good idea to predict discrete variables with linear regression. 

A simple type of classification problem is *binary classification*, where the target variable can take one of two values. It is common to denote the two levels of a binary variable by $\{0,1\}$ so we need our hypothesis function $0\leq h_{\theta}(x)\leq 1$. We construct this using the sigmoid form

\[
h_{\theta}(x) = \frac{1}{1+e^{-\theta^Tx}} = g(\theta^Tx)
\]

The function $h_{\theta}$ is interpreted as the probability that the target variable is 1 given $x$ and parametrized by $\theta$, and we denote the sigmoid function by $g$. 

\[
h_{\theta}(x) = \mathbf{P}(y=1|x;\theta)
\]

**Tip : ** A logistic regression problem has another parameter. Given a hypothesis function $h_{\theta}(x)$, what is the threshold for the prediction to be $y=1$ rather than $y=0$ ? A sensible boundary may be $h_{\theta}=0.5$, but in practice, this parameter needs to be optimised on the training set (or a validation set) to obtain the best possible predictions.  

The **decision boundary** is the hyper-surface defined by $\theta^Tx=0$ (corresponding to $h_{\theta}=0.5$) that is supposed to separate the two classes from each other. As $\theta\to\theta^*$ (the optimum), the decision boundary approaches the best possible separation between the two classes.

**Non-linear decision boundaries** are achieved through *feature engineering*, and including as features various powers and functions of the original $\{x\}$, as before. Arbitrarily complex decision boundaries are possible with this kind of feature engineering. 

### Cost function for classification

If the hypothesis function $\frac{1}{1+e^{-\theta^Tx}}$ is substituted into the quadratic error we used for linear regression, the resulting cost function $J(\theta)$ turns out not to be convex, and with lots of local optima that will make it impossible for gradient descent to find the global minimum.

The function of the error commonly used for logistic regression is 

  \[
  \text{ErrCost}(\theta,x^i) = \begin{cases} 
      -\text{log}(h_{\theta}(x^i)) & y^i= 1 \\
      -\text{log}(1-h_{\theta}(x^i)) &  y^i=0
   \end{cases}
\]
\[
J(\theta) = E_i[\text{ErrCost}(\theta,x^i)]
\]
It is clear that this function for the error imposes a very heavy penalty (which can be $\infty$) for a completely wrong prediction. This also grantees a convex $J(\theta)$ for logistic regression.

Since $y\in \{0,1\}$, we can write,
\[
J(\theta) = E_i[-y\text{log}(h_{\theta}(x^i))-(1-y)\text{log}(1-h_{\theta}(x^i))]
\]
This cost function follows from maximum likelihood estimation. A cool derivation to look up. The optimal values $\theta^*$ are obtained by gradient descent as before.

### Using other optimization algorithms

Gradient descent is not the only algorithm available to us. Given that we can compute $J(\theta)$ and $\frac{\partial}{\partial\theta_j}J(\theta)$, we can also use  

- Conjugate gradient descent  
- BFGS  
- L-BFGS  

etc. to compute the optimum. Most of them pick the learning rate $\alpha$ adaptively. They also converge faster. Looking up quickly what they do, is not a bad idea, but it is possible to use these optimization algorithms as black boxes without looking into the details. 

One must write an efficient function that can supply the value of $J(\theta)$ and its derivatives, and supply this function to one of Octave's in built optimization routines. Using these routines and more sophisticated optimization algorithms enables us to use logistic and linear regression on larger data sets.

### Multiclass classification

Simple logistic regression works well for binary classification, but what is a good way to generalize it to a problem with multiple categories for the target variable ? We use an idea called **one-vs-all classification**. So, if we have $C$ categories for the target variable $y\in \{ l_1\cdots l_C\}$, we create $C$ new training sets each of which has a binary target variable. In the $q^{th}$ training set the target variable is 1 only if $y=l_q$ and 0 otherwise. Now, we can train a logistic regression classifier for each of these $C$ sets, $h^1_{\theta}\cdots h^q_{\theta} \cdots h^C_{\theta}$, where $h^q_{\theta}(x)=\mathbf{P}(y=l_q|x;\theta)$ and the final prediction is given by 
\[y=\max_q h^{q}_{\theta}(x)\]

## Overfitting

Many learning algorithms - if not used carefully - build models that capture the behaviour of the training set as well as the *noise* that is inherent in any data set. In general, a very rigid (**biased** because it has encodes some strong assumptions that cannot be changed, e.g.. "the model is a straight line" is one such strong assumption) model will not be able to capture some important behaviour of the data. On the other hand a *very flexible* model (a very high order polynomial, for instance) will end up capturing a lot of noise from the data, and will not predict well when presented with new data which will have a different realization of randomness. Such a model is said to have high **variance** and this phenomenon is called **overfitting**. 

The link to the usual scientific idea about fitting is that given enough parameters, any data set can be fit perfectly. That does not mean that such a model will be good at predicting the result when new data is presented to it. However, with too few parameters - too simple/rigid/biased a model - the data cannot be properly explained. This is the **bias variance tradeoff**. 

Visualizing the data and the model can he helpful in diagnosis of over-fitting, but more importantly, good data visualization and exploration could help make appropriate model decisions to optimise predictive power despite the trade-off between bias and variance. 

If there is over-fitting, the following remedy could be tried :  

*Reduce the number of features*  
    - Manually select important features  
    - Model selection algorithms (dealt with later)  
*Regularization*  
    - Keep all features, but modify/reduce the values of parameters of the model  
    - Works well when there are a lot of features and each has low predictive power.  

### Regularization 

**Regularization** basically consists of imposing a **cost on complexity** of the model. So, a model with large number of parameters, or a model that is very flexible (models with low bias, high variance) are penalized with a term in the cost function. So, the cost function might become 
\[J_R(\theta) = J(\theta) + \underbrace{\lambda\cdot f_R(\theta)}_{\text{regularization term}}\]
where $\lambda$ is called the regularization parameter. Larger $\lambda$ imposes more of a cost on flexibility and makes the model more rigid (pushes the bias-variance trade-off toward more bias, less variance). Low $\lambda$ improves the fitting accuracy with increased model flexibility (pushes the bias variance trade-off toward less bias, more variance)

**Regularized linear regression**

One possible regularised cost function for linear regression would be a sum of parameter squares,
\[J_R(\theta) = \frac{1}{2m}\left[\sum_{i=1}^m (h_{\theta}(x^i)-y^i)^2 + \underbrace{\lambda\sum_j (\theta_j)^2}_{\text{regularization term}}\right],\]
then, the gradient descent algorithm for this cost function looks like
\[\text{repeat until convergence}\left[\theta_j :=\theta_j-\alpha\underbrace{\left(\frac{\lambda}{m}\theta_j + \frac{1}{m}\sum_i (x^i\theta-y^i)x^i_j\right)}_{=\frac{\partial}{\partial\theta_j}J_R(\theta)}\right]\]

The normal equation (derived in the same way as before) looks like
\[\theta = (X^TX-\lambda I)^{-1}X^Ty\]
The Identity matrix is used since here, we are treating all the $\theta$s in the same way. If - as in the lectures of Prof. Ng - one wants to leave $\theta_0$ (the bias term) out of the regularization process, then one use a matrix in which the first row of the identity matrix is replaced with all 0s.

For wide data $m<n$ (more features than examples) $X^TX$ is non invertible. But, the regularised version  $X^TX-\lambda I$ solves this issue too.

**Regularized logistic regression**

The regularized cost function for logistic regression follows along the same lines as for linear regression,
\[
J_R(\theta) = E_i[-y^i\text{log}(h_{\theta}(x^i))-(1-y^i)\text{log}(1-h_{\theta}(x^i))]+\frac{\lambda}{2m}\sum_j (\theta_j)^2
\]
Gradient descent too has the same form, but of course, the hypothesis function is the logistic function.

## The anatomy of supervised learning

For **any supervised machine learning problem**, the following are true :

1. There is a **training set**. This consists of data which we will use to teach the "machine" about the system we are interested in. We organize the training set such that each row of the data is an independent example, and each column is one feature, or predictor. We call this table $X$. The target variable - which is to be predicted - we call $y$. The problem is to identify a good function $y=h(x)$. If we are successful, then our function $h$ will correctly predict the target variable for data it has not seen, from a set commonly called the **test set**.

2. For a given problem and data, we start with some **hypothesis** about the form of the function $h$ parametrized by the parameter set $\theta$. In this scheme, our assumption is that $y=h_{\theta}(x)$. 

3. We define a **cost function** $J(\theta,X,y)$ that quantifies the error in prediction on the training set using $h_{\theta}$. A good cost function not only reflects the quality of the prediction, but is also well behaved and differentiable (since it will be fed to an optimization algorithm).

4. Now, the machine learning problem is reduced to an **optimization problem**. We need to find the set $\theta^*$ which minimises $J(\theta,X,y)$. Generally, we write a subroutine that takes in the training data points and a particular value of $\theta$ and returns the value and derivatives of $J$ with respect to the $\theta$s. This can be supplied to fast optimising routines found in any number of easily available libraries, and a good value of $\theta^*$ is found that minimises the cost function $J$.

5. Now, we have constructed a **model** for our data - $h_{\theta^*}$. When we have some new and unseen row of data $\hat{x}$, we can make a prediction by evaluating $h_{\theta^*}(\hat{x})$. How well a machine learning algorithm does, is judged by how predictions on an unseen test set where the correct $\hat{y}$ are known. So, the performance of the machine learning model is given by $J(\theta^*,\hat{X},\hat{y})$.


## Neural Networks

Logistic regression with feature engineering (higher order polynomial features etc) works well when there are small number of features. When there are a very large number of features already, even adding an exhaustive list of quadratic features becomes unreasonable, let alone higher orders. This makes logistic regression unsuitable for complex problems with large number of features. 

Even small pictures (for instance) have thousands and thousands of pixels, each of which has some values (intensity, colour) associated with it. This is a very large list of features. It is not reasonable to include all the higher order combinations of such a large feature set for an algorithm like logistic regression.

Neural Networks turn out to be a much better way to deal with this need for non linearity. One example of the power of neural networks is illustrated by the **one learning hypothesis** - the brain (a super complex neural network) does not do each learning task with a unique algorithm, but uses the same learning algorithm to do ALL the learning tasks it encounters. Then, it would make sense to mimic this learning algorithm to recreate such versatility in our machine learning systems. This is where neural networks excel. There are all kinds of cool experiments where the brain can learn to [interpret new sensory stimuli](https://thepsychologist.bps.org.uk/volume-25/edition-12/exotic-sensory-capabilities-humans), or learn to see with the audio cortex, etc.

A very good introduction to the fundamentals of neural networks is available [from M. Nielsen here](http://neuralnetworksanddeeplearning.com/chap1.html).

On our scheme, each neuron as several inputs $\{x\}$, and one output $h_{\theta}(x)$. Each neuron will typically implement a sigmoidal function (parametrized by $\theta$) on the inputs. As before,  
\[h_{\theta}(x) = g(\theta^Tx) = \frac{1}{1+e^{-\theta^Tx}}.\]

**Notation :** 

The inputs are **layer 1**, the last layer ($L$) is the **output layer** and all intermediate layers are **hidden layers**. The activation (computed output) of a neuron $i$ in layer $j$ is denoted by $a^j_i$. The weight of the output $x_p$ from a neuron $p$ in layer $j-1$ coming into neuron $i$ in layer $j$, is represented by $\Theta^{j-1}_{ip}$. So, for each set of connections from layer $j-1\to j$ we have the weight matrix $\Theta^{j-1}$. Now, the activation of all neurons in layer $j$ is given by the vector,
\[a^j = h(\Theta^{j-1}x),\]
where $h$ is the sigmoidal function as before.
By convention, a bias input $x_0$ present at each neuron. So, if there are $s_{j-1}$ neurons in layer $j-1$ and $s_j$ neurons in layer $j$, $\Theta^{j-1}$ is a $s_j\times (s_{j-1}+1)$ dimensional matrix. 

If we have one hidden layer with 3 neurons, and 3 inputs (plus a bias), and one output neuron, then our neural network is represented by,
\[\begin{bmatrix} x_0\\ x_1\\ x_2\\ x_3 \end{bmatrix} \to
\begin{bmatrix} a^2_0\\ a^2_1\\ a^2_2\\ a^2_3 \end{bmatrix} \to
h_{\theta}(x),
\]
where $h_{\Theta}(x)$ represents the computation of the output neuron on the inputs from the hidden layer. For this case, the output is,
\[
h_{\Theta}(x) = g\left(\Theta^2*\underbrace{\text{adBias}(\Theta^1*\text{adBias}(x^T))}_{\text{output of layer 2 }\{a^2\}}\right)
\]
where the function $\text{adBias}()$ increases dimension by 1 and adds the bias input $1$ for each layer, and $\Theta^1$, $\Theta^2$ are the weights to go from layer $1 \to 2$, and $2 \to \text{output}$ layer respectively.


This process of computing the output of the neural network by starting with the input layer is called **forward computation**. Each neuron is just a logistic regression unit, with the features being the outputs of neurons in the last layer. This means that each layer _learns_ what the best features are, to solve the given problem. This is what eliminates the need to include huge numbers of higher order terms as we would if were just doing logistic regression. 

The way neurons are linked up in an artificial neural network is known as the **architecture** of the neural network.

The output layer of a neural network can have more than one neuron. This enables multi-class classification, and we associate each class we are interested in with a vector, each component being the output of one output neuron. 


## Backpropagation 

The feed forward network with some parameters $\Theta$ described above can be thought of as some complicated hypothesis function that maps $x\to y$. To find the parameters that correspond to a good (predictive) mapping, we need to (as before) define a cost function and find the parameters that minimise it. 

### Cost function for neural network

Let the number of layers in a neural network be denoted by $L$ and the number of neurons in a layer $l$ be denoted by $s_l$. In terms of classification problems with $K$ classes, we will have $s_L=K$ neurons in the output layer. 

Clearly, we need a generalization of the cost function for logistic regression. Instead of one value, the network is outputting a $K$ dimensional vector. To reflect this we sum over all the outputs and the cost function can be written as,

\[J(\Theta)= -\frac{1}{m}\left[\underbrace{\sum_i}_{\text{examples}}\underbrace{\sum_k}_{\text{o/p}} y^i_k\text{log}(h_{\Theta}(x^i)_k)+(1-y^i_k)\text{log}\left(1-h_{\Theta}(x^i)_k\right)\right]+\underbrace{\frac{\lambda}{2m}\underbrace{\sum^{L-1}_l\sum^{s_l}_i\sum^{s_{l+1}}_j}_{\text{layers, weights}} (\Theta^l_{ji})^2}_{\text{regulatization term}}\]

In the cost function above, the bias parameters are not penalized in the regularization term, per convention. The parameters of the neural network are the $\Theta^l_{ji}$ each of which is the weight to go from neuron $i$ of layer $l$ to neuron $j$ of layer $l+1$. 


### Learning/optimizing the parameters of a neural network

As before, we need to search for the parameters $\Theta^*$ that minimise the cost function $J(\Theta)$. The algorithm used to do this for neural networks is called the **backpropagation** algorithm. As before, for particular values of $\{\Theta^l_{ij}\}$ we need to compute the value $J(\Theta)$ as well as all the derivatives $\frac{\partial}{\partial \Theta^l_{ij}}J(\Theta)$. 

To calculate the dependence of $J$ on the parameters, we use the chain rule. Consider the parameters $\Theta^{L-1}$ which feed the output layer,

\[
\begin{align}
\frac{\partial J(\Theta)}{\partial \Theta^{L-1}_{ji}} &= \frac{\partial J(\Theta)}{\partial h^L_j}
{\frac{\partial h^L_j}{\partial z^{L}_j}}
\underbrace{\frac{\partial z^{L}_j}{\partial \Theta^{L-1}_{ji}}}_{h^{L-1}_i} \\
&= \delta^L_j h^{L-1}_i.
\end{align}
\]
where $h^L_j$ denotes the activation of the $j^{th}$ neuron in the $L^{th}$ layer (in this case, the output layer), and the quantities below the under-braces follow from trivial differentiation. We defined $z^{L}_j = \Theta^{L-1}_j\text{adBias}(h^{L-1})$ and,
\[
\delta^L_j = \frac{\partial J(\Theta)}{\partial h^L_j}\frac{\partial h^L_j}{\partial (z^{L}_j)} = \frac{\partial J(\Theta)}{\partial h^L_j}g'(z^{L}_j)
\]

We would like to calculate the $\delta^{L-1}$s based on our knowledge of the $\delta^{L}$s. This is the essence of back-propagation, we compute the errors for earlier layers based on the error we compute at the output. What follows, is going to be a repeated application of uni and multivariate chain rules (see [this blog for another detailed explanation](http://adbrebs.github.io/Backpropagation-simply-explained/))

\[\begin{align}
\delta^{L-1}_j &= \frac{\partial J(\Theta)}{\partial h^{L-1}_j}\frac{\partial h^{L-1}_j}{\partial (z^{L-1}_j)} \\
&=  \frac{\partial J(\Theta)}{\partial z^{L-1}_j}\\
&= \sum_k\frac{\partial J(\Theta)}{\partial z^{L}_k}\frac{\partial z^{L}_k}{\partial z^{L-1}_j}\\
&= \sum_k \delta^{L}_k \frac{\partial z^{L}_k}{\partial z^{L-1}_j}\\
&= \sum_k \delta^{L}_k \frac{\partial z^{L}_k}{\partial h^{L-1}_j}\frac{\partial h^{L-1}_j}{\partial z^{L-1}_j}\\
&= g'(z^{L-1}_j)\sum_k \delta^L_k \Theta^{L-1}_{kj}
\end{align}\]

Then, the usual gradient descent algorithm tells us,

\[
\begin{align}
\Delta \Theta^{L-1}_{ji} &= -\alpha \frac{\partial J(\Theta)}{\partial \Theta^{L-1}_{ji}} \\
&= -\alpha \delta^L_j h^{L-1}_i.
\end{align}
\]

And, since we can calculate all the $\delta$s successively starting from the output layer, we can derive the corrections to all the parameters of the neural network.

**In matrix form : **

\[
\delta^{l-1} = g'(z^{l-1})\odot(\Theta^{l-1})^T\delta^l\\
\Delta \Theta^{l-1} = -\alpha (h^{l-1})^T\delta^l\\
h^l = g(\Theta^{l-1}\text{adBias}(h^{l-1}))
\]

It is implicit in the above discussion that this process is conducted for one training example at a time. So,

1. Using forward propagation, calculate the output $h_{\Theta}(x)$ for one training example.
2. Using the above prescription and the correct output $y$, calculate the correction to each parameter by successively calculating the $\delta$s, starting from the output layer. (be careful about the scaling of the regularization term, to update for each example, the regularization parameter $\lambda$ should be divided by $m$).
3. Repeat for every training example.
4. This yields $J(\Theta)$ and its derivatives. Supply these to a good optimisation algorithm, which will repeat steps 1-3 (back-propagation) to calculate the cost function and it's derivatives for each iteration.


**Tip :** In any implementation of back-propagation, it is a good idea to check that the algorithm is computing the right derivatives by implementing a simple numerical differentiation loop to check that back-propagation is giving the right values for derivatives. 

**Tip :** Network architecture.  
Number of i/p neurons - dimension of input vector,  
Number of o/p vector - number of classes,   
So, the questions remain, how many hidden layers and how many neurons should each one have ? The more hidden neurons the better, but of course larger networks are more computationally expensive. Usually, number of hidden units is or the order of dimension of input vector.



### Visualizing the innards of a neural net

For a dataset like [MNIST](http://yann.lecun.com/exdb/mnist/) of handwritten digit images, each 20x20 pixels in size, the first (input) layer has $s_1=400$. Thus, each neuron in the first hidden layer has $400+1$(bias) inputs going to it, one from each pixel. After the network has been trained, we can visualize the weights of these 400 inputs as $20\times 20$ images, and learn what aspect of the picture each of the neurons in the first hidden layer is looking for. Thus, in this case, each row of $\Theta^1$ can be visualized as an image representing the input to each of the neurons, similarly for subsequent layers. 

This sort of visualization tells us what elements of the picture each neuron is looking for (it will be activated/fire if that element is present), and thus it becomes clear how subsequent layers look for combinations of these basic elements, and so on. The power of deep learning and neural networks to learn at multiple levels of abstraction is neatly illustrated even in a dataset as seemingly simple as MNIST.

The most famous visualization of neural nets, of course, is [Google's inceptionism](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) blog where neural networks trained to detect various objects in pictures were used to generate images.. and representations of the objects the networks were supposed to detect came through in novel and interesting ways. 

## Measuring, improving predictive performance

Arbitrarily low training error is possible with an arbitrarily complex model. But, over-fitting the training data reduces prediction accuracy on new data. This is where **test sets** become useful. So, we randomly split our training set into two, say in a $0.7/0.3$ split. Then, we first learn on the training set by minimising $J(\theta)$, and compute the test set error $J_{test}(\theta)$. Then perhaps we can modify $\lambda$ to optimise $J_{test}(\theta)$. This mitigates the over-fitting problem.

### Train/Validation/Test sets

One way of choosing a good model is to take an ensemble of models (say, represented by different values of $\lambda$), train all of them, on the training set, and calculate test set error. Form the ensemble of trained models, we have chosen a model that best fits the test set. Essentially, we have used the test set to fit the extra parameter $\lambda$. This means that the performance on the test set is not indicative of performance on unseen data ! Hence, split data into three sets - Training/**Validation**/Test sets. 

Choose the model (trained on the training set) that optimises $J_{valid}(\theta)$ and then evaluate $J_{test}(\theta)$, which will then be an indicator of how well the chosen model will perform on new data.

## Machine learning diagnostics

How can one improve the performance of a predictive algorithm on test data if it is not satisfactory ? The following approaches can help -

1. **More training data**. in some settings, this could help. But, not always.  
2. **Check for overfitting**, and reduce number of features used.  
3. **Check for underfitting**, and see what other features can be included.  
4. **Feature engineering**. Check if including some functions of existing features improves performance. Sometimes, finding the right function of existing performance can radically improve performance.  
5. **Modify regularization parameter $\lambda$**.

But, which one should you do ? How to those a course of action ? This is where **ML diagnostics** enter the picture. 

### Bias Vs Variance

Consider the three errors $J_{train}$ on the training set, $J_{valid}$ on the validation set, and $J_{test}$ on the test set. We always have,

\[J_{train} < J_{valid} < J_{test}.\]

- When the regularization parameter $\lambda$ is very large, we are penalising the parameters of the model and **underfitting** the data (high bias) : **high $J_{train}$ and high $J_{valid}$**.  
- For very low values of $\lambda$ where our model might be **overfitting** the data (high variance)  : **very low $J_{train}$ and high $J_{valid}$**.

Plotting these errors against $\lambda$ (or another parameter that indicates model complexity) can be instructive. $J_{valid}$ has a minima at the optimal model complexity. That is the sweet spot one wants to hit.

**Tip :** While the cost function includes the regularization term, it is clear that to evaluate model performance, the regularization term is not relevant. So, it is not included in the error terms mentioned above. 

### Learning curves

Take small subsets of various sizes of the training set, and train a particular model on each of them. Plot the errors $J_{train}$ and $J_{valid}$ as a function of training set size $m$. For small $m$, $J_{train}$ is small (since it is easy to fit a small number of examples) while the $J_{valid}$ is large (since the model has not had much data to learn from). As $m$ grows, $J_{train}$ increases and $J_{valid}$ decreases.

In the **high bias** case, $J_{valid}$ does not decrease much with $m$, while $J_{train}$ will increase a lot with $m$ and end up close to $J_{valid}$ quite quickly. So, a high bias learning algorithm does not perform much better with lots more data. 

In the **high variance** case, $J_{train}$ will increase slowly with $m$. The validation error $J_{valid}$ will decrease slowly because of over-fitting, and for moderate $m$ there will be a big gap between $J_{train}$ and $J_{valid}$. But, over-fitting is reduced (and accuracy increased) as more data is added. So, the curves $J_{train}$ and $J_{valid}$ come closer as $m$ increases. 

So once one has run some of the diagnostics above (error vs $\lambda$ plots, error vs $m$ plots (learning curves)) etc, one can consider the possible courses of action we had mentioned earlier :

1. Getting more training examples helps fix high variance (reduces over-fitting).   
2. Reducing number of features also helps fix high variance.  
3. Adding features and feature engineering helps fix high bias issues.  
4. Increasing $\lambda$ fixes high variance, while decreasing $\lambda$ fixes high bias. 

Needless to say, small neural networks are prone to under-fitting. Large neural networks are prone to over-fitting, so regularization is important. Worth trying neural nets with different number of hidden layers and finding out which of them performs well on the validation sets. 

### Tips from Ng

- Start with a **simple algorithm** that is easy to implement.  
- Plot **learning curves** to diagnose over/under fitting and decide on course of action.  
- **Error analysis.** examine, plot etc the examples from the validation set that the algorithm failed on, and try to spot patterns or features that can be used to improve performance. 
- **Skewed classes :** when the overwhelming number of examples fall into one class. E.g.. faulty parts. Only 0.01% of parts might be faulty, so just marking everything as fine will lead to 99.9% correct classification, and yet, not a single faulty part will have been caught. Thus, for such cases, a different error metric is needed. 
- **Precision/recall :** calculate number of true positives, true negatives and false positives and false negatives. \[\text{Precision} = \frac{\text{True positives}}{\text{# Predicted positives}}=\frac{\text{True positives}}{\text{True positives+False positives}}\]
    \[\text{Recall}=\frac{\text{True positives}}{\text{# Actual positives}} = \frac{\text{True positives}}{\text{True positives+False negatives}}\]
- **Precision/Recall tradeoffs :** trade-off occurs because increasing precision means reducing number of false positives, so stringent criteria for predicting a positive. This will inevitably mean that the number of false negatives increase too, leading to lower recall. And it works the other way too, increasing recall leads to lower precision. 
- **F$_1$ score :** comparing precision/recall numbers. \[F_1 = 2\frac{PR}{P+R}\] makes for a good metric that ensures neither precision $P$ nor recall $R$ are too low, if the $F_1$ score is quite good. Choose the value of the threshold (for logistic regression, say) that maximises the $F_1$ score on the cross validation set.
- **When is lots of data worth it ?** Learning algorithms with large number off parameters (low bias) need large data sets to prevent over-fitting. Basically, we address the bias problem with a flexible and powerful learning algorithm and we address the variance problem with the massive data set. 
- Always worth asking and investigating if the problem is soluble at all, before investing in big data and machine learning.
- **Feature engineering matters more than specific learning algorithm used**. The amount of data, the type of features created, and skill in how the learning algorithm is used affects results a lot more than using this or that algorithm.

## Support Vector Machines

**An alternative view of logistic regression** :
remember, the hypothesis function of logistic regression for an input vector $x$ is,
 
\[
h_{\theta}(x)=\frac{1}{1+e^{-\theta^T x}} = g(z)
\]
where 
\[
z=\theta^T x.
\]
Intuitively, if $y=1$, $h_{\theta}\approx 1\implies z\gg 0$ and $y=0$, $h_{\theta}\approx 0\implies z\ll 0$.
Recall the cost function of logistic regression 
\[
-\left[y\cdot \text{log}(h_{\theta}(x))+(1-y)\cdot\text{log}(1-h_{\theta}(x)) \right].
\] 
For a particular example $(x,y)$ where $x$ is the input vector and $y$ is the output, suppose $y=1$.
Then, the cost function becomes 
\[
\text{ErrCost}(z|y=1)=-\text{log}\frac{1}{1+e^{-z}}.
\]
To make a support vector machine, we basically use a new cost function $\text{cost}_1$ that approximates this cost function with 2 straight line segments, while approximating 
\[
\text{ErrCost}(z|y=0)=\text{log}\left(1-\frac{1}{1+e^{-z}}\right)
\] 
with a different cost function $\text{cost}_0$ also consisting of 2 line segments. This yields a simpler, faster optimization problem.

In particular, $\text{cost}_1$ is a straight line with negative slope with an x intercept at 1. For $x\geq 1$, $\text{cost}_1=0$. On the other hand, $\text{cost}_0$ is a straight line with positive slope with an x intercept at -1. For $x\leq -1$, $\text{cost}_0=0$. 

The cost function for **logistic regression**
\[
J(\theta) = E_i[-y^i\text{log}(h_{\theta}(x^i))-(1-y^i)\text{log}(1-h_{\theta}(x^i))]+\frac{\lambda}{2m}\sum_j (\theta_j)^2
\]
while, for **support vector machines** the cost function is written as
\[
J(\theta)=C\cdot mE_i[ y^{(i)}\text{cost}_1(\theta^T x^{(i)}) +(1-y^{(i)})\text{cost}_0(\theta^T x^{(i)})]+\frac{1}{2}\sum_j (\theta_j)^2.
\]
Apart from the more approximate cost functions, the differences in the two cost functions are a matter of convention. For SVMs, the relative weights of the errors and the regularization term is controlled by the parameter C that is multiplied to the error term, rather than $\lambda$ multiplied to the regularization term as in the cost function for logistic regression. Also, in support vector machines the error and regularization term are not divided through by the number of examples. These changes should not - of course - change anything fundamental in the optimization procedure. 

Unlike logistic regression which gives a probability, the hypothesis function of an SVM is,
\[
h_{\theta}(x) = \begin{cases}
                1 & \theta^T x\geq 0\\
                0 & \theta^T x<0
                \end{cases}.
\]

**Large margin classifier limit**

Due to the form of the functions $\text{cost}_1$ and $\text{cost}_0$, if $y=1$, we want $\theta^T x\geq 1$ (not just $\geq 0$) and if $y=0$, we want $\theta^T x\leq -1$ (not just $<0$). In other words, the boundaries for the two classes are separated from each other, unlike in logistic regression. In practice, SVMs choose separators between cases that have larger margins to all classes. hence the name. This happens because of the optimization problem we have defined with the cost function and the definitions of the functions $\text{cost}_1$ and $\text{cost}_0$. In the limit $C\gg 1$, SVMs are equivalent to large margin classifiers. 

On the other hand, large margin classifiers can be very sensitive to outliers, SVMs do not suffer from this as long as the parameter $C$ is chosen wisely. 

when the classes are well separated, we can set error to 0. Hence, the optimisation function becomes

\[
\hat{\theta} = \arg \min_{\theta}\left(\frac{1}{2}\sum_{j=1}^n\theta_j^2 \right)\text{ such that }
\begin{cases}
\theta^Tx\geq 1 & \text{if } y^{(i)}=1 \\
\theta^Tx\leq -1 & \text{if } y^{(i)}=0. 
\end{cases}
\]

### Kernels - adapting SVMs for non linear decision boundaries

One way to get non linear boundaries is to include polynomial features and treat those as new predictors, as we discussed for logistic regression. But, for complex problems, higher order polynomials are not really a good choice and can be very computationally expensive to include all necessary features. 

A better way to pick features is using **landmarks**. Certain points are identified in the space of features as being in some way specially significant to the problem at hand, and proximity (using some notion of distance or similarity) to these points is used to compute further features. For instance, given landmarks $l^{(1)},l^{(2)},l^{(3)}$ in the space of features, we can define one feature to be 

\[
f_1 = \exp\left(-\frac{\norm{x-l^{(1)}}^2}{2\sigma^2}\right)
\]

The specific similarity functions used are called **kernels**. In this case, we are using a *gaussian* kernel for $f_1$. It is clear that the Gaussian kernel falls away from the landmark at a rate determined by $\sigma$ and has a value of 1 at the landmark, and 0 infinitely far from the landmark. For classification problems, it is clear how choosing landmarks at estimated or intuitive or known centres of classes would be a good choice. 

In fact, given a limited number of $m$ training examples, each training example can be a landmark leading to a new feature vector $f=\{f_1, f_2,...f_m\}$. So, for a training example $x^{(i)}$, we have the feature vector $f^{(i)} = \{f^{(i)}_1,f^{(i)}_2,f^{(i)}_3....f^{(i)}_m\}$ where $f^{(i)}_j$ is the similarity measure (given by the kernel) of the $i^{th}$ training example from the $j^{th}$ landmark (which, in this case, is the $j^{th}$ training example.. so $f^{(k)}_k = 1$ in this particular case). 

**Definition of SVM with kernels**

- Hypothesis : given $x$, compute the features $f\in \mathbb{R}^{m+1}$. Parameters $\theta\in\mathbb{R}^{m+1}$, predict $y=1$ if $\theta^Tf\geq 0$.   
- Training : 
\[
\hat\theta = \arg \min_{\theta}\left[ C\cdot \sum_{i=1}^m \left( y^{(i)}\text{cost}_1(\theta^Tf^{(i)}) + (1-y^{(i)})\text{cost}_0(\theta^Tf^{(i)})\right) + \frac{1}{2}\sum_{j=1}^m \theta_j^2\right]
\]

Intuition on over, under fitting :

- **Large** $C\implies$ low bias, high variance, while **small** $C\implies$ high bias low variance.  
- **Large** $\sigma\implies$ smoothly varying features, high bias low variance, while **small** $\sigma\implies$ sharp features, low bias, high variance.

**Using SVMs in practice**

While the algorithms used to solve the optimization problem are available in many software libraries, we do have to make some choices in order to use an SVM to solve our problem.

- A value of the parameter C.  
- An appropriate kernel, and parameters involved therein.   

There are many good choices of kernels depending on the problem and structure of the data, but a valid kernel needs to satisfy [Mercer's theorem](https://www.quora.com/What-is-an-intuitive-explanation-of-Mercers-Theorem) in order to be compatible with the optimisation procedure for SVM implementations. 

**Tips from Ng** :

Let $n$ be number of features, $m$ be number of training examples.

- if $n\geq m$, use logistic regression or SVM with no kernel.  
- if $n\leq 10^3$ and $m\leq 10^4$, use SVM with Gaussian kernel.  
- if $n\leq 10^3$ and $m\geq 10^4$ add features using landmarks and use logistic regression or SVM without kernels.  

Neural networks will work well for most of these regimes, but will take a lot longer to train. One advantage of SVMs is that the optimisation problem is a convex problem, which means that the we are likely to end up close to a global optimum. Unlike in other algorithms, we don't have to worry about landing up in local optima. 

## Unsupervised learning

Unsupervised learning algorithms find structure in unlabelled datasets. for instance, clustering. Clustering problems occur in may contexts :    
- market segmentation  
- organizing computing clusters  
- astronomical data analysis  
- social network analysis  

### K-Means clustering algorithm

1. Randomly initialize $K$ cluster centroids $\{\mu_1, ... \mu_k...\mu_K\} \in \mathbb{R}^n$(if we want to cluster the data into n clusters) in the feature space of the data set.  
2. Assign each data point $i$ to the cluster $c^{(i)}$ with the closest cluster centroid $\mu_{c^{(i)}}$.   
3. Compute the mean of the data points assigned to each cluster, and move the cluster centroid to that location. (update $\{\mu_1, ... \mu_k...\mu_K\}$)  
4. Repeat the cluster assignment of step 2. (update cluster assignments $c^{(i)}$ for each data point $i$)  
5. Repeat steps 2-4 until the cluster centroids don't move (much) any more.   

*how cool will it be to visualize a K-Means run on flat, sperical, toroidal etc geometries ! does it converge on a mobius strip ? on a sphere ?*

**Optimization objective function** for the K-Means algorithm : 
we have $m$ data points, and $K$ clusters. Then, 
\[
J(c^{(1)},...c^{(m)},\mu_1,...\mu_K) = \frac{1}{m}\sum_{i=1}^{m}||x^{(i)}-\mu_{c^{(i)}}||
\]

This function is also called the **distortion**. The K-Means algorithm minimises this function. Clearly, the cluster assignment step is minimising the distances from data points to the cluster centroid of the clusters they are assigned to, and then, the re calculation of the cluster centroids again reduces the distance by moving the cluster centroids to the centre of mass. 

**Randomly initializing cluster centroids** :   
- clearly, $K<m$.  
- randomly select $K$ training examples, and set the $K$ centroids to these examples.   
- K-Means can end up at different solutions depending on initial centroid initialization and it can end up in bad local optima.   
- we can try multiple random initializations and choose the one that converges to the best (smallest cost function) solution.  

**How to choose number of clusters ?** : 
Most popularly, do some data exploration and visualization and choose the number of clusters by hand. But, this may be genuinely hard, or unclear. 

*Elbow method *- Plot the cost function against the number of clusters chosen. It often turns out that until a certain number of clusters chosen, the distortion decreases rapidly, and after that point goes down very slowly (forming an elbow). Then choose the number at the bend of the elbow. 

However, sometimes the distortion goes down smoothly with number of clusters (this is a lot more common). In this case, *optimise the number of clusters $K$ for the ultimate purpose for which the clustering is being done*. E.g.. if we are clustering a population into sizes to manufacture t-shirts, then we can do the clustering for several values of $K$, and see how much business sense it makes to have those clusters, with the population segmented that way, in terms of cluster sizes, t-shirt fits, etc. 

### Dimensionality reduction

What is it good for ?  
- **data compression** : basically, finding a more efficient representation of the data in a smaller number of dimensions.  
- **visualization** : if dimensionality can be reduced to 3, or even better, 2 dimensions, then, structure in the data that might otherwise be difficult to see, might be easily visualized. 

### Principal Component Analysis

Essentially, PCA searches for a lower dimensional surface such that the sum of squares for the distance from the data points to the surface (projection error) is minimised. It is important to normalize and scale the features before PCA (so that the distances in different directions in the feature space are comparable). 

To reduce from $n$ dimensions to $k$ dimensions, we want to find the $k$ vectors $u^{(1)}..u^{(k)}$ onto which to project the data such that the projection error is minimized. 

**The algorithm :**  
1. always start with mean normalization and feature scaling.  
2. compute the $n\times n$ covariance matrix $\Sigma = \frac{1}{m}\sum_{i=1}^{n}(x^{(i)})(x^{(i)})^T = \frac{1}{m}X^T X$ or $\Sigma = \frac{1}{m}X^TX$.  
3. compute the eigenvectors of $\Sigma$, using the singular value decomposition function `svd` (normally, the `eig` function would be used, but for covariance matrices (the way they are constructed) the singular value decomposition gives the same eigenvectors) see [this page](https://math.stackexchange.com/questions/320220/intuitively-what-is-the-difference-between-eigendecomposition-and-singular-valu) for some excellent intuitive explanations.  
4. in octave, `[U,S,V] = svd(Sigma)` and `U` has the eigen vectors. To reduce $n$ dimensions to $k$, just take the first $k$ columns of the matrix `U`. $U_r$ is $n\times k$.   
5. the new $m\times k$ data $Z = X^TU_r$ where $X$ is the original $n$ dimensional data with $m$ examples.  

If this sort of thing is to be used for data compression, clearly, we need to be able to go back to the $n$ dimensional space, with some loss of information due to the compression procedure. This is just the $m\times n$ matrix $X_{\text{approx}} = ZU_r^T$.

**How many principle components should I keep ?** :  
- average squared projection error = $\frac{1}{m}\sum_{i}^{m}||x^{(i)}-x_{\text{approx}}^{(i)}||^2$  
- total variation in the data = $\frac{1}{m}\sum_{i=1}^m ||x^{(i)}||^2$  
- choose $k$ to be the smallest value such that 99% of the variance is retained,   
\[
\frac{\frac{1}{m}\sum_{i}^{m}||x^{(i)}-x_{\text{approx}}^{(i)}||^2}{\frac{1}{m}\sum_{i=1}^m ||x^{(i)}||^2}\leq 0.01
\]
- the way to check this, is using the matrix `S` from the `[U,S,V] = svd(Sigma)`. `S` is a diagonal square matrix $S_{ii}$. for a given $k$, we have,
\[
\frac{\frac{1}{m}\sum_{i}^{m}||x^{(i)}-x_{\text{approx}}^{(i)}||^2}{\frac{1}{m}\sum_{i=1}^m ||x^{(i)}||^2} = 1 - \frac{\sum_{i=1}^k S_{ii}}{\sum_{i=1}^n S_{ii}},
\]
so, with just one run of the `svd`, we can find the value of $k$ we need. [Here is](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca) an excellent account of the relationship between singular value decomposition and PCA.

**Using PCA to speedup a learning algorithm :** essentially, learning on very high dimensional data is hard. With PCA, we can reduce the dimensionality and thus, in due to trivial computational reasons (fewer numbers to crunch !) get any algorithm to run faster. Clearly, we must apply PCA on training set, find the mapping (the learned matrix $U_r$) and apply the same mapping to cross validation and test data. 

Ng says PCA should be used as a part of pre-processing for ML algorithms *only if* running with the raw data does not work. 

### Anomaly detection

**Problem definition** : Given a dataset of "normal" examples $X$, is a new example $x_{test}$ anomalous ?
Usually, this is approached by building a probability distribution $P_X$ over $X$ and computing $P_X(X_test)$ and then, if $P_X(X_test)<\epsilon$ for some sensible $\epsilon$, then we might classify $x_{test}$ is anomalous. For example,   

- **fraud detection** : measure user characteristics (login times, typing speed etc.) on a website, and flag users that are behaving unusually.   
- **manufacturing** : measure features for each machine/product, and when there is a machine/product whose $P(x)$ is very small, it can be flagged for further analysis/maintenance.   

All of this sounds like a job for [kernel density estimation](https://arxiv.org/abs/1704.03924), does it not :)

**The algorithm :**

- Training set: $\{x^{(1)},....,x^{(m)}\}$ each $x\in \mathbb{R^n}$.  
- each $p(x^{(i)}) = \prod_{j=1}^n p(x^{(i)}_j;\mu_j, \sigma^2_j)$ is a product of independent Gaussian.  
- Choose features $\{x_j\}$ that might be indicative of anomalous examples  
- Fit parameters $\{\mu_j, \sigma^2_j\}$ for each feature  
- given new example, compute   
\[
p(x) = \prod_{j=1}^n p(x_j;\mu_j,\sigma^2_j)  
\]
- anomaly if $p(x)<\epsilon$.

**Real number evaluation** while evaluating an algorithm, always best to have a method that returns a number.. which allows us to gauge how good the algorithms is. 

for anomaly detection, assume $y=1$ is anomalous, $y=0$ is non anomalous. Then, several metrics are possible, $F_1$ score, for instance. The hyper-parameter $\epsilon$ should be chosen on the cross validation set, and then the test set performance should be indicative of real performance. 

Why not use supervised learning ?

- very few anomalies in training set, so algorithm cannot really know all the possible anomalies, making supervised learning useless to detect new anomalies  
- if there are lots of anomalies, then supervised learning has a chance. but, for rare positives/anomalies.. best to go with anomaly detection, since the learning algorithm cannot learn much from the anomalous examples.   

Since a lot of density estimation algorithms are based on Gaussian distributions, it is best to transform all features so that they look vaguely Gaussian. 

If $p(x)$ is similar for normal and anomalous examples.. then adding a feature which helps identify anomalies will help. If $p(x)$ is too small both for normal and anomalous examples, removing features with large variation might help.

Clearly, the assumption of independence of features is a strong one. A true multivariate distribution will do a much better job of anomaly detection. In that case, the problem does reduce to multivariate KDE. 

### Recommender systems

Recommender systems are massively useful systems that directly add to the profits of many companies. Fundamentally, recommenders are market lubricants, facilitating exchange of information to improve number of exchanges made. 

One of the "big ideas" of machine learning is the notion of **automatically learning features** instead of hand coding them in, and recommender systems are a good setting to show this. 

**Example 1:** Predicting movie ratings.  
Index $i$ denotes movie, $j$ denotes person. Then, for each pair $(i,j)$ we either have a movie rating $y(i,j)$ and a flag $r(i,j)=1$ (if a movie $i$ has been watched by the person $j$), or the flag $r(i,j)=0$, if person $j$ has not watched movie $i$. We want to predict the ratings $y$ for the cases when $r(i,j)=0$. let $n_m$ is number of movies, $n_u$ is number of users. 

**Content based recommenders :**  Suppose that for each movie, we have features $x_1$ which measures how romantic a movie is, and $x_2$ which measures how much of an action movie it is. In general, there could be lots of such features based on the content of the movie. For each movie $i$, we have a feature vector $x^{(i)}$. We could now treat predicting the ratings for each user as a regression problem. In the linear regression case, for user $j$, we have a parameter vector $\theta^{(j)}$. Once we learn these parameter vectors $\{\theta^{(j)}\}$, for a movie with feature vector $x^{(i)}$, the predicted rating is just $y(i,j) = \theta^{(j)}\cdot x^{(i)}$. The parameters $\theta^{(j)}$ is learnt on the basis of linear regression on the movies that user $j$ has rated, for each user $j$.

Of course, a lot of the time, we might not have content based features for various movies. Hence, **collaborative filtering**. Here, we know nothing about the content of our movies, but, we do know something about our users. Each user $j$ just tells us $\theta^{(j)}$ via some survey. Then, based on available ratings $y(i,j)$ when $r(i,j)=1$, we can infer the feature vectors $x^{(i)}$, since we have $y(i,j) = \theta^{(j)}\cdot x^{(i)}$ using linear regression, where the $x^{(i)}$ are the parameters. Once the $x^{(i)}$ are known, the $\theta$ vectors for new users can be estimated based on their ratings, as before. 

This suggests an iterative process :  
- guess random $\theta$s  
- infer $x$ via known ratings  
- infer $\theta$ based on $x$ and ratings  
- repeat until reasonable convergence.  

But, there is a more efficient algorithm that does not need to iterate. Instead, just treat $x$s and $\theta$s as one set of parameters $\{...x^{(i)}......\theta^{(j)}...\}$. The modified optimization objective is 
\[
J(..x^{(i)}......\theta^{(j)}...) = \frac{1}{2}\sum_{(i,j):r(i,j)=1}\left(\left(\theta^{(j)}\right)^Tx^{(i)}-y(i,j)\right)^2 + \frac{1}{2}\sum_i\sum_k \left( x_k^{(i)} \right)^2 + \frac{1}{2}\sum_j\sum_k \left( \theta_k^{(j)} \right)^2
\]
where we must minimise over all $x$s and $\theta$s. 

The **collaborative filtering algorithm** is :   
1. Initialize the parameters $\{...x^{(i)}......\theta^{(j)}...\}$ to small random values  
2. Minimise $J(..x^{(i)}......\theta^{(j)}...)$ over all parameters using gradient descent.  
3. For a user with parameters $\theta$ and a movie with features $x$, the rating is $\theta\cdot x$.   

**Vectorized collaborative filtering**

\[Y_{(i,j)} = \theta^{(j)}\cdot x^{(i)}\]
\[X_{(i,k)} = x^{(i)}_k\]
\[\Theta_{(j,k)} = \theta^{(j)}_k\]
\[Y = X\Theta^T\]

This is called **low ranked matrix factorization** because $Y$ is [low ranked.](https://en.wikipedia.org/wiki/Rank_(linear_algebra)) 

Once we know features related to movies $x^(i)$ then finding movies related to a given movie $x^{(i_0)}$, one can just calculate the distances $||x^{(i_0)}-x^{(i)}||$ and pick a few movies with the lowest distances. 

In general, it's best to regularize the means for various known quantities. 

## Large scale machine learning

when starting with a big data set, **always** first try with small subsets, and plot the learning curves ($J_{train}, J_{CV}$ vs $m$) to ensure that your learning algorithm has a large variance for small $m$.

### Gradient descent with large datasets

Recall the gradient descent update rule -
\[
\theta_j :=\theta_j - \alpha \frac{1}{m}\sum_i^m (h_{\theta}(x^i)-y^i)x^i_j
\]
When $m$ is very large, each step of the gradient descent algorithm requires summing over a huge $m$, and this is computationally hugely expensive and time consuming. 

**Stochastic gradient descent**

The usual gradient descent is called *batch gradient descent*, when all training examples are used to update the parameters in the $\frac{1}{m}\sum_i^m (h_{\theta}(x^i)-y^i)x^i_j$ term (which reflects the derivative of the cost function $J_{train}(\theta) = \frac{1}{2m}\sum_{i=1}^m (h_{\theta}(x^i)-y^i)$).

For stochastic gradient descent :  
1. define a cost function for one training example $cost(\theta, i) = \frac{1}{2}(h_{\theta}(x^i)-y^i)^2$  
2. shuffle the order of training dataset  
3. Repeat until reasonable results (between 1-10 times), 
\[
\text{for i=1..m} \\
\left\{
\text{for j=i..n} \\
\theta_j := \theta_j-\alpha((h_{\theta}(x^i)-y^i)x^i_j)
\right\}
\]
This does not really converge, but it ends up with parameters in the vicinity of the global minimum. In exchange for very significant computational savings.  

**Mini-batch gradient descent**

Batch gradient descent - use all $m$ examples in each update.   
Stochastic gradient descent - use 1 example in each update.  
Mini-batch gradient descent - use $b$ examples in each iteration.  

\[
\text{for i=1..}\frac{m}{b} \\
\left\{
\text{for j=i..n} \\
\theta_j := \theta_j-\alpha\frac{1}{b}\sum_{k=1}^{b-1}((h_{\theta}(x^{(i-1)b+k})-y^{(i-1)b+k})x^{(i-1)b+k}_j)
\right\}
\]

this gives better performance than stochastic gradient if we have a very good vectorized implementation. Of course, this is same as batch gradient descent if $b=m$. 

**Tips for ensuring stochastic gradient descent is working**

- during learning compute $cost(\theta, i) = \frac{1}{2}(h_{\theta}(x^i)-y^i)^2$ before updating $\theta$ with that training example.  
- every 1000 steps (say) plot $cost(\theta, i)$ averaged over the last 1000 examples.  
- this plots should slowly get better as more examples are processed.   
- this might suggest using a smaller learning rate, since the oscillations around the minimum will now be smaller  
- slowly decrease learning rate to get stochastic gradient descent to converge $\alpha = \frac{c_1}{\text{iter}+c_2}$, but then there are two more hyper-parameters that need to be fiddled with  

### Online learning : continuous data stream  

An increasingly common setting, since a lot of websites and other companies collect very large amounts of data in real time and need to use it in real time. Now, we discard the notion of a fixed training set. An example comes in, we update our model with the data, and abandon the data and just keep the updated model. If there is a small number of users, it might make sense to store all the data.. but for huge data volumes, it makes sense to learn from incoming traffic and let your model learn continuously. 

This has the advantage of letting your website/business adapt to changing user preferences. 

### Map-reduce and parallelism 

Some data problems are too large to handle on one machine. Such problems are usually tackled with clusters of computers, and map-reduce is a frame work to parallelize work over several machines. Can handle problems far larger than stochastic gradient descent. 

If there are $m$ training examples, and there are $q$ machines to run these on, then $m/q$ are sent to each machine, and each machine computes 
\[
t^q_j = \sum_k^{(m/q)} (h_{\theta}(x^k)-y^k)x_j^k
\]
then, we can compute the update for batch gradient descent 
\[
\theta_j := \theta_j -\alpha\frac{1}{m}\sum_q t_j^q
\]

and (ignoring overheads) we can get a maximum speed-up of $q$ times. This basically, parallelizes the calculation of the sum involved in gradient descent updates. 

The **key question :** can the learning algorithm be expressed as a sum of some functions over the training set ? if so, map-reduce can help.

For instance, for many optimization algorithms, we need to provide them with cost functions (thats one sum) and gradient (another sum), so for large data sets, map-reduce can parallelize these sums and pass these values to the optimization algorithm.  

On multi-core machines, map-reduce can already help by paralleling. But, in such cases, vectorized implementations along with a very good, parallelized linear algebra library will take care of this. Hadoop has this system under the hood.

## General lessons from a case study (photo-OCR)

- define a **pipeline** for the ML problem. The photo-OCR pipeline :   
    1. text detection  
    2. character segmentation  
    3. character classification   
- sliding window classification :  
    1. if we have say 50px $\times$ 50px images of an object, we obtain lots of 50 $\times$ 50 images without the object and train a supervised learning classifier.  
    2. given an image, we slide a 50$times$50 window over the image and run the classifier at each step  
    3. how much the sliding window moves, is determined by the stride parameter  
    4. then, do this for a larger windows (by taking larger bits of the image and compressing down to 50 $times$) and run the classifier over the image (to detect the object at different scales).  
    5. coalesce nearby positive responses into common rectangles using an expansion operator (classify nearby negative pixels to positive too, up to a certain distance).    
- Artificial data synthesis :  
    - creating data from scratch : for say, text detection, take random text,m transform it into many random fonts, and paste each piece onto a random background. this is some work, but good synthetic data creation can lead to an unlimited supply of labeled data to help solve your problem.  
    - amplifying a small training set : for each element in the training set, add various warpings, colours, backgrounds, noise etc. with insight and thought, it can lead to a much amplified training set. for different problems, of course the distortions added will be different. For instance, for audio, we can add different background noises etc. The distortions introduced should be representative of the sorts of distortions that might come up in the test set.  
- Before setting out to get more data -  
    1. is our algorithm low bias ? plot learning curves  
    2. "How much work would it be to get 10x as much data ?" if one can brainstorm ones way to lots more data with a few days of work, large improvements in performance can be expected. mechanical turk is an option.    
- Ceiling analysis : what to work on next  
    - one of the most valuable resources is your time spent working on system.  
    - pick a single real number evaluation metric for the over all system and measure it  
    - now, fix the test set with labels that let one module do it's job with 100% accuracy, now measure overall system accuracy  
    - do this for each module in turn starting with the most upstream component, and work on the module that creates the largest impact on the overall system  
