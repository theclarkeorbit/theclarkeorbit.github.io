---
title: "greta playground"
output:
  html_document:
    df_print: paged
---
A first foray into probabilistic programming with Greta


## Models and modelling

Much of science - physical and social - is devoted to positing mechanisms that explain how the data we observe are generated. In a classic example, after [Tycho Brahe](https://en.wikipedia.org/wiki/Tycho_Brahe) made detailed observations of planetary motion ([here](http://www.pafko.com/tycho/observe.html) is data on mars), [Johannes Kepler posited laws](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion) of planetary motion that _explained_ how this data were generated. Effectively, **modelling** is the art of constructing data generators that help us understand and predict. 

**Statistical modelling** is one class of models that aims to construct - given some observed data - the probability distribution from which the data were drawn. That is, given a sample of data, a statistical model is a hypothesis about how this data were generated. In practice, this happens in two steps :  
- constructing a hypothesis, or a model $H$ parametrized by some parameters $\theta$  
- finding (_inferring_) the most suitable parameters $\theta$ given the observed data

What parameters are "most suitable" is defined by the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function) that quantifies how probable the observed data set is, for a given hypothesis parametrized by some particular parameters $H_{\theta}$. Understandably, we want to find parameters such that the observed data is the most likely, this is called [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

Since any but the simplest models are analytically intractable (i.e., the maximum of the likelihood function needs to be evaluated numerically) it makes sense to construct general rules and syntax to easily define and quickly infer the parameters of statistical models. This is the field of probabilistic programming. 

## Probabilistic programming

The probabilistic programming language (PPL) has two tasks :  

1. be able to construct a useful class of statistical models  
2. be able to infer the parameters of this class of models given some observed data.  

As has been explained in this [excellent paper introducing the PPL Edward](https://www.reddit.com/r/deeplearning/comments/846wb6/the_paper_that_introduces_the_edward_ppl_by/) that is based on Python and [Tensorflow](https://www.tensorflow.org/), some PPLs restrict the class of models they allow in order to optimize the inference algorithm, while other emphasize expressiveness and sacrifice performance of the inference algorithms. Modern PPLs like [Edward](http://edwardlib.org/), [Pyro](https://eng.uber.com/pyro/), and the R based [Greta](https://greta-dev.github.io/greta/index.html) use the robust infrastructure (hardware and software) that was first developed in the context of deep learning and thus ensure scalability and performance while being expressive. 

### The tensor and the computational graph

The fundamental data structure of this group of languages is the [tensor](https://en.wikipedia.org/wiki/Tensor) which is just a multidimensional array. Data, model parameters, samples from distributions are all stored in tensors. All the manipulations that go into the construction of the output tensor constitute the computational graph (see [this](http://colah.github.io/posts/2015-08-Backprop/) for an exceptionally clear exposition of the concept) associated with that tensor.  

Data and parameter tensors are inputs to the computational graph. In the context of deep learning, "training" consists of the following steps :
1. Randomly initializing the parameter tensors  
2. Computing the output  
3. Measuring the error compared to the real/desired output  
4. Tweaking the parameter tensors to reduce the error.  
The algorithm that does this is called [back propagation](https://en.wikipedia.org/wiki/Backpropagation).
Thus, the objective in deep learning or machine learning is to obtain the **best values** of the parameters given some data.

The objective of probabilistic modelling is subtly different. The objective here is to obtain the **distribution** (called **posterior distribution**) of parameters given the data. If we denote the data by $D$, [Bayes theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) connects (for a particular hypothesis about how the data were generated $H$), the likelihood of the data given some parameters $P(D|\theta,H)$, our prior expectations about how the parameters are distributed $P(\theta)$ and the posterior distribution of the parameters themselves $P(\theta|D,H)$ :
\[
P(\theta|D,H) = \frac{P(D|\theta,H)P(\theta)}{P(D)}.
\]

The priors $P(\theta)$ do not depend on the data and encode "domain knowledge" while and the probability of the data set $P(D)$ is (typically) a high dimensional integral given by
\[
P(D|H) = \int P(D,\theta|H)d\theta.
\]

Intuitively, we want to compute the most likely parameters given the data, i.e. we want to maximize $P(\theta|D,H)$. While maximizing the likelihood can give us the estimates of the "most likely parameters" (in the limit of infinite data), computing the full distribution $P(\theta|D,H)$ involves the computation of the difficult integral for $P(D|H)$.

### Bayesian prediction and MCMC

Prediction in this framework is also fundamentally different from typical machine learning model. The probability of a new data point $d$,
\[
P(d|D,H) = \int P(d|\theta,H)P(\theta|D,H)d\theta,
\]
which consists of the expectation value of the new data point over the whole distribution of parameters given the observed data (the posterior distribution calculated obtained from the solution to the inference problem), instead of a value calculated by plugging in the "learned parameter values" into the machine learning model. 

The integrals needed for inference ($P(D|H) = \int P(D,\theta|H)d\theta$ as well as prediction $P(d|D,H) = \int P(d|\theta,H)P(\theta|D,H)d\theta$ are over the parameter space which can be very high dimensional. This, Markov Chain Monte Carlo methods are used to approximate these integrals. [This](https://www.reddit.com/r/deeplearning/comments/8487xg/very_good_introduction_to_hamiltonian_monte_carlo/) is an excellent overview of modern Hamiltonian Monte Carlo methods while [this](https://www.reddit.com/r/MachineLearning/comments/84fobk/superb_overview_and_motivation_for_monte_carlo/?ref=share&ref_source=link) provides wonderful perspective from the dawn of the field. Both papers are long but eminently readable and highly recommended. 

Clearly then, along with the computational graph to define models, a PPL needs a good MCMC algorithm (or another inference algorithm) to compute the high dimensional integrals needed to infer as well as perform a prediction on a general probabilistic model. 

A broad overview of Bayesian machine learning is available [here (PDF)](http://mlg.eng.cam.ac.uk/zoubin/talks/mit12csail.pdf) and [here](http://fastml.com/bayesian-machine-learning/)



















