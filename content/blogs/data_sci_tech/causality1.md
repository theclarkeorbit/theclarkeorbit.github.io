---
title: "Climbing Pearl's Ladder of Causation"
date: 1 October 2025
output:
  html_document:
    df_print: paged
---



This article heavily uses the `quartets` package and the [paper that introduces it](https://www.tandfonline.com/doi/full/10.1080/26939169.2023.2276446) but there is also another [paper](https://sites.stat.columbia.edu/gelman/research/published/causal_quartet_second_revision.pdf)(pdf) and associated [package](https://github.com/jhullman/causalQuartet) on a very similar theme. For analysing DAGs, we rely heavily on the wonderful `dagitty` package, and we visualize using `ggdag`. 


``` r
# Load required packages
library(quartets)      # Our example datasets
library(dagitty)       # For DAG manipulation and testing
library(tidyverse)     # For data manipulation and basic plotting
library(ggthemes)      # For theme_tufte
library(patchwork) 
library(ggdag)

# Set a theme for all plots
theme_set(theme_tufte(base_size = 11))
```

If you are totally new to all this causal stuff, do read [The Book of Why](https://dl.acm.org/doi/10.5555/3238230) (no, really).

## The wind/rudder problem

Consider a dynamic system consisting of three variables: wind speed $W$, rudder angle $R$, and boat heading $D$ (direction). An observer on shore measures $R$ and $D$ continuously over time, with the goal of understanding whether rudder adjustments causally affect the boat's trajectory. The wind $W$, however, remains unobserved.

The data-generating process works as follows. The wind $W$ exerts a direct causal influence on the boat's heading: $W \rightarrow D$. A skilled sailor observes the wind and adjusts the rudder to compensate, creating a second causal relationship: $W \rightarrow R$. The rudder also directly affects heading through hydrodynamic forces: $R \rightarrow D$. Crucially, the sailor's adjustment strategy is systematic: rudder angle is set to approximately counteract the wind's effect, meaning $R \approx -\alpha W$ for some coefficient $\alpha > 0$.

After collecting observational data, the observer computes $\text{cor}(R, D)$ and finds it to be approximately zero. The naive conclusion follows: rudder angle has no relationship with boat direction, therefore the rudder is ineffective.

The error stems from **endogeneity**. In econometrics and causal inference, a variable is called endogenous when it is correlated with the error term or, equivalently, when it is determined by other variables in the system that also affect the outcome. Here, the rudder angle $R$ is endogenous because it is not set independently of the other factors affecting direction, $R$ is systematically determined by $W$ (via the sailor working to keep the boat's heading constant), which is itself a common cause of both $R$ and $D$. 

This creates what Pearl calls **confounding**: the naive correlation $\text{cor}(R, D)$ confounds the true causal effect of $R$ on $D$ with the spurious correlation induced by their common cause $W$. This and other seemingly intractable issues like [Simpson's paradox](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2343788) (association between two variables changes sign when conditioned on a third) are all unravelled once we encode our knowledge of the world into causal relationships between variables and test how valid these are in light of data. 

At the end of this article, you should be able to do a basic causal analysis on an appropriate dataset with dagitty and have an intuition about what it is you are saying when you estimate direct and indirect causal effects.

## The data.

Let's load the causal quartet and take a first look. These four datasets were carefully constructed to have identical correlations between x and y, but as we'll discover, they tell four completely different causal stories.


``` r
data("causal_quartet")
sample_n(causal_quartet,10)
```

```
## # A tibble: 10 × 6
##    dataset        exposure outcome covariate     u1     u2
##    <chr>             <dbl>   <dbl>     <dbl>  <dbl>  <dbl>
##  1 (4) M-Bias       0.600   0.867      2.54   0.364 -0.362
##  2 (4) M-Bias       2.45   -0.143      5.24   0.544 -1.35 
##  3 (4) M-Bias       0.784   2.39       5.68   0.586  0.681
##  4 (1) Collider     0.691   1.95       2.77  NA     NA    
##  5 (4) M-Bias      -0.377  -1.87     -12.9   -1.58  -1.22 
##  6 (1) Collider     0.261  -0.668     -0.749 NA     NA    
##  7 (1) Collider     0.0472  0.0305    -0.438 NA     NA    
##  8 (4) M-Bias      -1.57   -3.08      -6.70  -0.760 -0.840
##  9 (3) Mediator     1.11    2.70       2.36  NA     NA    
## 10 (2) Confounder   0.406   0.451      0.655 NA     NA
```

``` r
causal_quartet |>
  group_by(dataset) |>
  summarise(
    n = n(),
    mean_x = mean(exposure),
    mean_y = mean(outcome),
    sd_x = sd(exposure),
    sd_y = sd(outcome),
    cor_xy = cor(exposure, outcome)
  )
```

```
## # A tibble: 4 × 7
##   dataset            n  mean_x  mean_y  sd_x  sd_y cor_xy
##   <chr>          <int>   <dbl>   <dbl> <dbl> <dbl>  <dbl>
## 1 (1) Collider     100 -0.146  -0.0136  1.05  1.37  0.769
## 2 (2) Confounder   100 -0.0291 -0.0357  1.30  1.77  0.735
## 3 (3) Mediator     100 -0.0317 -0.0392  1.03  1.72  0.594
## 4 (4) M-Bias       100  0.136  -0.0809  1.48  2.06  0.724
```

``` r
ggplot(causal_quartet, aes(x = exposure, y = outcome)) +
  geom_point(alpha = 0.6, size = 2) +
  facet_wrap(~ dataset, labeller = label_both) +
  labs(
    title = "The Causal Quartet: Identical Statistics, Different Stories",
    subtitle = "Four datasets with the same correlation but different causal structures",
    x = "X",
    y = "Y"
  ) +
  theme(
    strip.text = element_text(face = "bold", size = 10)
  )
```

![center](/figures/causality1/load-data-1.png)

All four datasets have nearly identical means, standard deviations, and correlations. If we only looked at these summary statistics, we might think they're the same data. They look similar, but the underlying causal mechanisms are completely different. To see why, we need to understand Pearl's ladder.

## The ladder

Judea Pearl posits that causal reasoning exists on three distinct levels. Each rung requires more sstructure than the one below it, and each rung enables us to answer questions that are impossible at lower levels.

**Rung 1: association** - The world of pure observation. We ask "what is?" We observe patterns and correlations. This is the domain of traditional statistics. We can compute things like P(Y|X), the probability of Y given that we observe X. Remarkably, even at this rung of the ladder, causal assumptions made explicit can - sometimes - be falsified because different causal relationships result in different implications for conditional distributions between variables.

**Rung 2: intervention** - The world of action and experiments. We ask "what if I do?" We imagine actively manipulating one variable and seeing what happens to another. This is the domain of randomized controlled trials and policy evaluation. We compute things like P(Y|do(X)), the probability of Y if we *force* X to take a particular value - in other words, severe X from this causes and set it to a value we choose. In practice, this is often hard to do so the ability to reason about interventions without making them (by modifying the DAG and working out the implications) is very useful.

**Rung 3: counterfactuals** - The world of imagination and retrospection. We ask "what would have been?" We look at what actually happened and imagine how it would have been different under alternative circumstances. This is the domain of individual-level causal attribution, regret, and responsibility. We compute things like "what would Y have been for this specific individual if X had been different?". Apart from the data and the DAG, we also need to add structural equations (functional relationships) that define how each arrow of the DAG is actualized. Then, we infer something about these functions based on the real data  - treatment A failed for Patient 2 tells us something - and thus we can attempt to answer a question like "Would Patient 2 have recovered with treatment B?". 

Each rung requires adding something to what we had before. Let's climb the ladder step by step, using the causal quartet as our guide.

## Rung 1: association 

At the first rung, we're observing the world and looking for patterns. We have data, and we can compute conditional probabilities, correlations, and statistical associations. But we cannot distinguish between causation and correlation. All we can say is "when I see X, I tend to see Y."

### What we add: DAG structure

To work at this level, we add a directed acyclic graph (DAG) that represents possible conditional independence relationships. The DAG is just a hypothesis about the structure of dependencies between variables. At this level, the edges don't necessarily mean "causes" - they just mean "statistically dependent."

The four datasets in the causal quartet actually correspond to four simple DAG structures. Let's take a look.


``` r
# Dataset 1: Collider (X causes Y, Both cause Z)
dag1 <- dagitty('dag {
  exposure -> outcome
  exposure -> covariate
  outcome -> covariate
}')

# Dataset 2: Common cause / Confounder (Z causes both X and Y)
dag2 <- dagitty('dag {
  covariate -> exposure
  covariate -> outcome
  exposure -> outcome
}')

# Dataset 3: Mediation / Chain (X causes Z, Z causes Y)
dag3 <- dagitty('dag {
  exposure -> covariate -> outcome
}')

# Dataset 4: M-bias (X and Y both cause Z)
dag4 <- dagitty('dag {
  u1 -> covariate
  u1 -> exposure -> outcome
  u2 -> covariate
  u2 -> outcome
}')

# Plot them side by side
dag1 |> 
  tidy_dagitty() |> 
  ggplot(aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_point(color = "#C4B5A0", size = 20, alpha = 0.4) +
  geom_dag_text(color = "#D6604D", size = 3.0) + geom_dag_edges(edge_colour = "#C4B5A0") +
  ggtitle("1. Collider") +
  theme_dag() -> p1

dag2 |> 
  tidy_dagitty() |> 
  ggplot(aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_point(color = "#C4B5A0", size = 20, alpha = 0.4) +
  geom_dag_text(color = "#D6604D", size = 3.0) + geom_dag_edges(edge_colour = "#C4B5A0") +
  ggtitle("2. Confounder") +
  theme_dag() -> p2

dag3 |> 
  tidy_dagitty() |> 
  ggplot(aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_point(color = "#C4B5A0", size = 20, alpha = 0.4) +
  geom_dag_text(color = "#D6604D", size = 3.0) + geom_dag_edges(edge_colour = "#C4B5A0") +
  ggtitle("3. Mediator") +
  theme_dag() -> p3

dag4 |> 
  tidy_dagitty() |> 
  ggplot(aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_point(color = "#C4B5A0", size = 20, alpha = 0.4) +
  geom_dag_text(color = "#D6604D", size = 3.0) + geom_dag_edges(edge_colour = "#C4B5A0") +
  ggtitle("4. M-bias") +
  theme_dag() -> p4

p1 + p2 + p3 + p4 + 
  plot_layout(ncol = 2, nrow = 2)
```

![center](/figures/causality1/define dags-1.png)

While the headline stats are the same, the 4 data sets represent very different causal structures, seen above. 

### Testing DAGs against data

In a typical situation, the DAG is an explicit set of assumptions about the world generated by the scientist, and these need to be tested against the data. The key insight of the first rung of causation is that different DAG structures imply different patterns of conditional independence. Even though x and y are correlated in all four datasets, they have different relationships when we condition on the third variable z.

We can use the **d-separation criterion** (two variables are independent given your controls if no path between them can transmit information) to derive testable implications. The `dagitty` package does this automatically for us.


``` r
# What conditional independencies does each DAG imply?
cat("DAG 1 (Collider) implies:\n")
```

```
## DAG 1 (Collider) implies:
```

``` r
print(impliedConditionalIndependencies(dag1))

cat("DAG 2 (Confounder) implies:\n")
```

```
## DAG 2 (Confounder) implies:
```

``` r
print(impliedConditionalIndependencies(dag2))

cat("DAG 3 (Mediation) implies:\n")
```

```
## DAG 3 (Mediation) implies:
```

``` r
print(impliedConditionalIndependencies(dag3))
```

```
## exps _||_ otcm | cvrt
```

``` r
cat("DAG 4 (M-bias) implies:\n")
```

```
## DAG 4 (M-bias) implies:
```

``` r
print(impliedConditionalIndependencies(dag4))
```

```
## cvrt _||_ exps | u1
## cvrt _||_ otcm | exps, u2
## cvrt _||_ otcm | u1, u2
## exps _||_ u2
## otcm _||_ u1 | exps
## u1 _||_ u2
```

We see that DAGs 1,2 can't be falsified at Rung 1, since there are no conditional implications we can test (and which could potentially be false).

Before we start testing our DAG hypotheses with data, we need to understand what we're actually testing and how to interpret the results. This is subtle because the logic is **inverted from typical hypothesis testing**. The testing procedure works as follows:

1. Extract all conditional independence implications from the DAG using d-separation
2. For each implication of the form $X \perp Y \mid Z$ (for example):
   - Regress $X$ on $Z$ to get residuals $r_X$
   - Regress $Y$ on $Z$ to get residuals $r_Y$  
   - Test whether $\text{cor}(r_X, r_Y) = 0$
3. Return a p-value for each test

Interpreting the results:

The null hypothesis $H_{\phi}$ is that the variables ARE conditionally independent (as the DAG claims). Therefore:

- **High p-value (> 0.05)**: We fail to reject $H_{\phi}$. The conditional independence holds in the data. The DAG's prediction is not contradicted. The test passes!
- **Low p-value (< 0.05)**: We reject $H_{\phi}$. The conditional independence is violated. The DAG's prediction is contradicted. The test fails.


Passing all tests does not *prove* the DAG is correct. Multiple different DAG structures can imply the same set of conditional independencies (these are called "Markov equivalent" DAGs). What we can do is *falsify* DAG structures that make predictions contradicted by the data. This is the essence of rung 1: we can use conditional independence testing to rule out impossible causal structures, but we cannot definitively prove which structure is correct from observational data alone.


``` r
# Test the fork structure against dataset 3 (should fail)
data3 <- causal_quartet |> 
  filter(dataset == "(3) Mediator") |> 
  select(exposure, outcome, covariate, u1, u2)
# Dataset 3: Mediation / Chain (X causes Z, Z causes Y)
dag3_wrong <- dagitty('dag {
  exposure -> outcome
  exposure -> covariate
  
}')
print(impliedConditionalIndependencies(dag3_wrong))
```

```
## cvrt _||_ otcm | exps
```

``` r
cat("Testing fork structure against dataset 3 (wrong structure):\n\n")
```

```
## Testing fork structure against dataset 3 (wrong structure):
```

``` r
results3_wrong <- localTests(dag3_wrong, data3, type = "cis")
print(results3_wrong)
```

```
##                        estimate      p.value      2.5%     97.5%
## cvrt _||_ otcm | exps 0.7605758 1.452368e-22 0.6732402 0.8750805
```

``` r
# Visualize the test results
plotLocalTestResults(results3_wrong)
```

![center](/figures/causality1/test-wrong-dag-1.png)

The test fails! Dataset 3 is not consistent with a fork structure. Instead, it follows a chain structure where x causes z and z causes y. Let's verify that the correct structure fits.


``` r
# Test the chain structure against dataset 3
cat("Testing chain structure (x -> z -> y) against dataset 3:\n\n")
```

```
## Testing chain structure (x -> z -> y) against dataset 3:
```

``` r
results3 <- localTests(dag3, data3, type = "cis")
print(results3)
```

```
##                          estimate p.value      2.5%     97.5%
## exps _||_ otcm | cvrt 0.002031441 0.98412 -0.195478 0.1993846
```

``` r
plotLocalTestResults(results3)
```

![center](/figures/causality1/test-dag3-1.png)

The chain structure fits dataset 3...

### Limitations of rung 1

"Correlation is not causation" is only the the beginnign of the story. We can use conditional independence testing to falsify causal structures that are inconsistent with the data.

But we cannot prove that any particular structure is correct, we can only show that some structures are inconsistent with the data. Moreover, we cannot distinguish between correlation and causation just from observational data. The edges in our DAGs could represent predictive relationships rather than causal ones.

To go further, we need to climb to the second rung.

## Rung 2: intervention

At the second rung, we move from passive observation to active manipulation. We imagine or actually perform interventions where we force a variable to take a particular value and observe what happens to other variables. This is the world of experiments, policies, and "what if I do?" questions. In pragmatic terms, we estimate the actual causal effect of changing a treatment on the outcome. 

### What we add: causal interpretation

The key addition at rung 2 is semantic: we now interpret the edges in our DAG as representing direct causal relationships, not just statistical dependencies. When we write $x \rightarrow y$, we mean "x directly causes y," not just "x predicts y."

We add a new mathematical operator: **do(X = x)**, which represents forcing X to take the value x. This is different from observing X = x. When we intervene, we perform graph surgery by removing all incoming edges to X, reflecting the fact that we've overridden X's normal causes.

### Estimating direct and total causal effects

Here we estimate causal effects by asking: "What would happen to the outcome if we *intervened* to set exposure to a  specific value?"

In Pearl's do-calculus notation, we write this as $\mathbb{E}[Y \mid do(X = x)]$ - the expected value of $Y$ when we set $X$ to $x$ (rather than just observe $X = x$). We target $\mathbb{E}[Y \mid do(X = x)]$. The total effect (ATE) is
$$
\tau = \mathbb{E}[Y \mid do(X = 1)] - \mathbb{E}[Y \mid do(X = 0)].
$$
A direct effect can be defined as a controlled direct effect
$$
\mathrm{CDE}(m) = \mathbb{E}[Y \mid do(X = 1, M = m)] - \mathbb{E}[Y \mid do(X = 0, M = m)],
$$
or as a natural direct effect (which requires stronger assumptions).

So, given a DAG, which variables should one control for to estimate $\mathbb{E}[Y \mid do(X)]$ from observational data? This is where the adjustment formula comes in, under certain conditions (backdoor criterion satisfied - set of variables blocks all backdoor paths from exposure to outcome and contains no descendants of exposure),

$$\mathbb{E}[Y \mid do(X = x)] = \sum_z \mathbb{E}[Y \mid X = x, Z = z] \cdot P(Z = z)$$

In practice with regression, if we adjust for the right $Z$, the coefficient on $X$ estimates the causal effect.

The `daggity` package has several helpful functions:

1. The `paths` function lists all paths from exposure to outcome and tells us if they are open or not:

``` r
p4
```

![center](/figures/causality1/unnamed-chunk-1-1.png)

``` r
paths(dag4, "exposure", "outcome")
```

```
## $paths
## [1] "exposure -> outcome"                         
## [2] "exposure <- u1 -> covariate <- u2 -> outcome"
## 
## $open
## [1]  TRUE FALSE
```
2. The `adjustmentSets` function tells us what to control for when estimating direct and total effets:

``` r
adjustmentSets(dag4, exposure = "exposure", outcome = "outcome",
                             effect = "direct", type = "minimal")
```

```
##  {}
```

``` r
adjustmentSets(dag4, exposure = "exposure", outcome = "outcome",
                             effect = "total", type = "minimal")
```

```
##  {}
```
which tells us that we don't need to control for anything in DAG-4 to estimate the total and direct effects. Moreover, since the only backdoor path is closed, the total and direct effects are going to be the same for this DAG. 

``` r
data4 <- causal_quartet |> 
  filter(dataset == "(4) M-Bias") |> 
  select(exposure, outcome, covariate, u1, u2)
lm(outcome ~ exposure, data4)
```

```
## 
## Call:
## lm(formula = outcome ~ exposure, data = data4)
## 
## Coefficients:
## (Intercept)     exposure  
##     -0.2175       1.0041
```
Lets see this for the mediator (DAG-3) (where we know the direct effect should be 0, since there is no arrow going from exposure to outcome):

``` r
p3
```

![center](/figures/causality1/unnamed-chunk-4-1.png)

``` r
adjustmentSets(dag3, exposure = "exposure", outcome = "outcome",
                             effect = "direct", type = "minimal")
```

```
## { covariate }
```

``` r
adjustmentSets(dag3, exposure = "exposure", outcome = "outcome",
                             effect = "total", type = "minimal")
```

```
##  {}
```
So, to estimate the direct effect we need to condition on the covariate:

``` r
lm(outcome ~ covariate+exposure, data3)
```

```
## 
## Call:
## lm(formula = outcome ~ covariate + exposure, data = data3)
## 
## Coefficients:
## (Intercept)    covariate     exposure  
##    0.014874     1.067466     0.002474
```
and we see that the direct effect of exposure on outcome is ~0. 
For the total effect, we don't need to control for anything:

``` r
lm(outcome ~ exposure, data3)
```

```
## 
## Call:
## lm(formula = outcome ~ exposure, data = data3)
## 
## Coefficients:
## (Intercept)     exposure  
##   -0.007654     0.995610
```
### So much linear regression!

While we have used `lm()` throughout, the DAG-based adjustment strategy (what to control for) remains valid regardless of the  statistical method. The DAG tells you what to adjust for, the statistical method determines how to adjust for it.

When relationships between variables arenon-linear, methods like [splines](https://noamross.github.io/gams-in-r-course/) or [generalized additive models (GAMs)](https://m-clark.github.io/generalized-additive-models/)  via the [`mgcv` package](https://cran.r-project.org/web/packages/mgcv/index.html) can be used.  These allow smooth, flexible relationships while remaining interpretable - one can visualize exactly how covariates affect the outcome. For example: 
`gam(outcome ~ exposure + s(covariate), data = data)` estimates the causal effect of the exposure while relaxing the.

Beyond controlling directly in a regression, there are other techniques like [propensity score methods](https://academic.oup.com/ejcts/article/53/6/1112/4978231) 
(matching or weighting by the probability of treatment) via packages like  [`MatchIt`](https://kosukeimai.github.io/MatchIt/) or [`WeightIt`](https://ngreifer.github.io/WeightIt/). For a "best of both worlds"  approach, [doubly robust methods](https://cran.r-project.org/web/packages/drtmle/vignettes/using_drtmle.html) combine outcome modeling with propensity scores—you get the correct answer if either model is right (see [`AIPW`](https://yqzhong7.github.io/AIPW/) package). This relies on balancing the groups (treated/untreated) on covariates by matching similar entities or reweighting the sample, so comparing outcomes is like comparing a randomized trial.

When treatment effects vary across individuals, [causal forests](https://grf-labs.github.io/grf/articles/grf.html)  from the [`grf` package](https://grf-labs.github.io/grf/) use machine learning to estimate personalized treatment effects non-parametrically. See [Athey & Wager (2019)](https://arxiv.org/abs/1902.07409) for the methodology, or the [excellent documentation](https://grf-labs.github.io/grf/articles/grf_guide.html) that comes with the `grf` package

All these methods relax parametric assumptions (linearity, functional form) but cannot escape the fundamental causal  assumptions—namely, that your DAG is correct and all confounders are measured.  As Judea Pearl emphasizes: ["no causes in, no causes out"](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf)—no statistical technique can create causal knowledge from purely observational data if the DAG is wrong. The fanciest machine learning model with the wrong adjustment set will give worse answers than simple linear regression with the right one.

### The power and limits of rung 2

At rung 2, we can answer policy questions and predict the effects of interventions. We can distinguish between genuine causal effects and spurious correlations due to confounding. This is very useful for science and decision-making.

We can tell the average effect of intervening on x in the population: $P(Y|do(X))$ tells us what would happen on average if we forced X for everyone. But we cannot tell you what would have happened for a specific individual who actually experienced one value of X. We cannot answer "what if things had been different for this person?" 

To answer that question, we need to climb to the third rung.

## Rung 3: counterfactuals

At the third rung, we move from population-level predictions to individual-level explanations. We look at what actually happened to a specific person or instance and ask "what would have been different if...?" This is the domain of regret, responsibility, attribution, and retrospection.

### What we add: functional mechanisms

The key addition at rung 3 is the specification of functional equations that describe how each variable is generated from its causes. We move from a DAG (which just shows which variables affect which) to a Structural Causal Model (SCM), which specifies the mechanisms.

For each variable, we write:
$$X_i = f_i(\text{Parents}(X_i), U_i)$$

where $f_i$ is a function and $U_i$ represents all the unmeasured factors (noise, randomness, individual variation) that affect $X_i$.

This is a big step up in specificity. At rung 2, we only needed to know the graph structure. At rung 3, we need to know the actual functional form of how causes produce effects.

### What we can do: individual counterfactuals

Let's work with dataset 3 (the chain structure) since it has a clear causal mechanism we can estimate. The data was actually generated from a specific structural model. Let's estimate that model and use it to answer counterfactual questions.


``` r
# Estimate the structural equations
# x -> z: z = a*x + u_z
# z -> y: y = b*z + u_y

model_x_to_z <- lm(covariate ~ exposure, data = data3)
model_z_to_y <- lm(outcome ~ covariate, data = data3)

cat("Structural equations:\n")
```

```
## Structural equations:
```

``` r
cat("covariate = ", round(coef(model_x_to_z)[1], 3), " + ", 
    round(coef(model_x_to_z)[2], 3), "*exposure + u_covariate\n", sep = "")
```

```
## covariate = -0.021 + 0.93*exposure + u_covariate
```

``` r
cat("outcome = ", round(coef(model_z_to_y)[1], 3), " + ", 
    round(coef(model_z_to_y)[2], 3), "*covariate + u_outcome\n\n", sep = "")
```

```
## outcome = 0.015 + 1.069*covariate + u_outcome
```

Now suppose we observe a specific individual with exposure = 1.5, covariate = 2.0, outcome = 3.5. We want to answer: "What would outcome have been for this individual if exposure had been 2.5 instead?"

This is a counterfactual question. To answer it, we use Pearl's three-step procedure:

1. Abduction: Given what we observed, infer the values of the unobserved noise terms $U_{covariate}$ and $U_{outcome}$ for this specific individual
2. Action: Modify the structural equations to reflect the intervention (set exposure = 2.5)
3. Prediction: Compute what outcome would have been using the modified equations and the inferred noise terms


``` r
# Observed values for one individual
observed_exposure <- 1.5
observed_covariate <- 2.0
observed_outcome <- 3.5

# Step 1: Abduction - infer the noise terms
u_covariate <- observed_covariate - predict(model_x_to_z, newdata = tibble(exposure = observed_exposure))
u_outcome <- observed_outcome - predict(model_z_to_y, newdata = tibble(covariate = observed_covariate))

cat("Step 1 - Abduction (infer noise terms):\n")
```

```
## Step 1 - Abduction (infer noise terms):
```

``` r
cat("u_covariate =", round(u_covariate, 3), "\n")
```

```
## u_covariate = 0.626
```

``` r
cat("u_outcome =", round(u_outcome, 3), "\n\n")
```

```
## u_outcome = 1.348
```

``` r
# Step 2: Action - set x to counterfactual value
counterfactual_exposure <- 2.5

# Step 3: Prediction - compute counterfactual y
# First, compute what z would have been
counterfactual_covariate <- predict(model_x_to_z, 
                            newdata = tibble(exposure = counterfactual_exposure)) + u_covariate

# Then, compute what y would have been
counterfactual_outcome <- predict(model_z_to_y, 
                           newdata = tibble(covariate = counterfactual_covariate)) + u_outcome

cat("Step 2 - Action (set x =", counterfactual_exposure, ")\n\n")
```

```
## Step 2 - Action (set x = 2.5 )
```

``` r
cat("Step 3 - Prediction:\n")
```

```
## Step 3 - Prediction:
```

``` r
cat("Counterfactual z would have been:", round(counterfactual_covariate, 3), "\n")
```

```
## Counterfactual z would have been: 2.93
```

``` r
cat("Counterfactual y would have been:", round(counterfactual_outcome, 3), "\n\n")
```

```
## Counterfactual y would have been: 4.494
```

``` r
cat("Summary:\n")
```

```
## Summary:
```

``` r
cat("Observed: x =", observed_exposure, ", y =", observed_outcome, "\n")
```

```
## Observed: x = 1.5 , y = 3.5
```

``` r
cat("Counterfactual: if x had been", counterfactual_exposure, 
    ", y would have been", round(counterfactual_outcome, 3), "\n")
```

```
## Counterfactual: if x had been 2.5 , y would have been 4.494
```

``` r
cat("Effect for this individual:", 
    round(counterfactual_outcome - observed_outcome, 3), "\n")
```

```
## Effect for this individual: 0.994
```

### The Connection to Bayesian Inference

What we did in step 1 (abduction) looks a lot like Bayesian inference. We observed some data and used it to infer the values of unobserved variables (the noise terms). In a Bayesian framework, we would treat the noise terms as latent variables and compute their posterior distribution given the observed data. Then, to compute counterfactuals, we would sample from this posterior and simulate what would have happened under different interventions.

This is exactly the approach taken in [Richard McElreath's "Statistical Rethinking"](https://www.youtube.com/playlist?list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus), where he shows how to use Bayesian inference to compute counterfactuals in structural causal models. The generative model (the SCM) becomes the inference target, and counterfactual reasoning becomes a form of posterior prediction under modified models.

### The Power and Limits of Rung 3

At rung 3, we can answer questions about individual instances and attribute causation at the finest grain. We can reason about responsibility, regret, and explanation. We can say not just "smoking causes cancer on average" but "Bob's cancer was caused by his smoking."

But this power comes at a price. We need much more information than at lower rungs. Specifically, we need:

1. The correct causal graph (rung 2 requirement)
2. The functional form of how causes produce effects
3. The distribution of unobserved factors

These requirements are often difficult or impossible to satisfy with observational data alone. We typically need strong assumptions about functional forms (linearity, additivity, etc.) or additional data from experiments.

## Returning to the Rudder Paradox

Let's generate some data that mimics the rudder situation and see how each rung of the ladder handles it.


``` r
# Simulate the rudder situation
set.seed(123)
n <- 500

# Generate data
# W = wind (unobserved!)
# R = rudder position (sailor adjusts to counteract wind)
# D = direction (affected by both wind and rudder)

W <- rnorm(n, mean = 0, sd = 1)  # Wind is unobserved
R <- -0.9 * W + rnorm(n, 0, 0.1)  # Rudder counteracts wind
D <- 0.5 * W + 0.5 * R + rnorm(n, 0, 0.1)  # Direction depends on both

rudder_data <- tibble(
  W = W,  # We'll pretend we don't observe this
  R = R,
  D = D
)

# The true causal structure
rudder_dag <- dagitty('dag {
  W -> R
  W -> D
  R -> D
}')

rudder_dag |> 
  tidy_dagitty() |> 
  ggplot(aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_point(color = "#C4B5A0", size = 20, alpha = 0.4) +
  geom_dag_text(color = "#D6604D", size = 3.0) + geom_dag_edges(edge_colour = "#C4B5A0") +
  ggtitle(paste("Rudder problem adjustment set ", adjustmentSets(rudder_dag, exposure = "R", outcome = "D", effect = "total", type = "minimal"))) +
  theme_dag()
```

![center](/figures/causality1/rudder-setup-1.png)

### Rung 1 Analysis

At rung 1, we only have observational data on R and D (we can't see the wind W). Let's look at their correlation.


``` r
# Compute correlation between rudder and direction
cor_RD <- cor(rudder_data$R, rudder_data$D)

cat("Correlation between rudder (R) and direction (D):", 
    round(cor_RD, 3), "\n")
```

```
## Correlation between rudder (R) and direction (D): -0.361
```

``` r
# Visualize
rudder_data |>
  ggplot(aes(x = R, y = D)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(
    title = "The Rudder Paradox",
    subtitle = paste0("Correlation = ", round(cor_RD, 3), 
                     " - Rudder appears useless!"),
    x = "Rudder Position",
    y = "Boat Direction"
  )
```

![center](/figures/causality1/rudder-rung1-1.png)

The correlation is near zero! At rung 1, we would conclude (incorrectly) that the rudder doesn't affect the boat's direction. The problem is confounding by the unobserved wind W. The wind affects both R and D in opposite ways, misleading us when we don't account for it.

### Rung 2 Analysis: Intervention Reveals Truth

At rung 2, we ask: what if we *intervene* to set the rudder at a specific position, overriding the sailor's control?

To answer this using the adjustment formula, we would need to condition on W (the wind). But W is unobserved, so we can't apply the standard backdoor adjustment. However, if we did have access to W, we could correctly estimate the causal effect.


``` r
# If we could observe W, we could adjust for it
model_adjusted <- lm(D ~ R + W, data = rudder_data)

cat("True causal effect of rudder on direction (adjusting for wind):\n")
```

```
## True causal effect of rudder on direction (adjusting for wind):
```

``` r
print(coef(model_adjusted)["R"])
```

```
##         R 
## 0.5531443
```

``` r
# Compare to the naive (wrong) estimate
model_naive <- lm(D ~ R, data = rudder_data)
cat("\nNaive estimate (not adjusting for wind):\n")
```

```
## 
## Naive estimate (not adjusting for wind):
```

``` r
print(coef(model_naive)["R"])
```

```
##           R 
## -0.05092029
```

When we properly adjust for the confounding wind, we see that the rudder has a substantial causal effect (around 0.5) on direction. The naive analysis severely underestimates this effect because it doesn't account for the confounding.

In a real experiment, we could reveal this by randomizing the rudder position. If we forced the rudder to different positions (breaking the W → R link), we would see the true causal effect.
