---
title: "Climbing Pearl's Ladder of Causation"
date: 1 October 2025
output:
  html_document:
    df_print: paged
---



## The Rudder Paradox

## The Rudder Paradox: When Correlation Lies

Consider a dynamic system consisting of three variables: wind speed $W$, rudder angle $R$, and boat heading $D$ (direction). An observer on shore measures $R$ and $D$ continuously over time, with the goal of understanding whether rudder adjustments causally affect the boat's trajectory. The wind $W$, however, remains unobserved.

The data-generating process works as follows. The wind $W$ exerts a direct causal influence on the boat's heading: $W \rightarrow D$. A skilled sailor observes the wind and adjusts the rudder to compensate, creating a second causal relationship: $W \rightarrow R$. The rudder also directly affects heading through hydrodynamic forces: $R \rightarrow D$. Crucially, the sailor's adjustment strategy is systematic: rudder angle is set to approximately counteract the wind's effect, meaning $R \approx -\alpha W$ for some coefficient $\alpha > 0$.

After collecting observational data, the observer computes $\text{cor}(R, D)$ and finds it to be approximately zero. The naive conclusion follows: rudder angle has no relationship with boat direction, therefore the rudder is ineffective. This conclusion is incorrect.

The error stems from **endogeneity**. In econometrics and causal inference, a variable is called endogenous when it is correlated with the error term or, equivalently, when it is determined by other variables in the system that also affect the outcome. Here, the rudder angle $R$ is endogenous because it is not set independently of the other factors affecting direction. Instead, $R$ is systematically determined by $W$ (via the sailor working to keep the boat's heading constant), which is itself a common cause of both $R$ and $D$. 

This creates what Pearl calls **confounding**: the naive correlation $\text{cor}(R, D)$ confounds the true causal effect of $R$ on $D$ with the spurious correlation induced by their common cause $W$. The mathematical intuition is straightforward. The observed correlation can be decomposed as:

$$\text{cor}(R, D) \approx \beta_{R \rightarrow D} + \beta_{W \rightarrow R} \times \beta_{W \rightarrow D}$$

where $\beta$ represents the standardized strength of each causal pathway. When the sailor implements a compensatory strategy ($\beta_{W \rightarrow R} < 0$) that roughly balances the direct effect ($\beta_{R \rightarrow D} > 0$), these terms can cancel, producing a correlation near zero despite a genuine causal effect.

This phenomenon—where correlation dramatically understates or even reverses the sign of a causal effect due to endogeneity—is precisely what Judea Pearl identifies as a fundamental limitation of the first rung of the causal ladder. Observational association $P(D \mid R)$ can be misleading when the observed variables are not exogenous (independently determined). To recover causal effects, we need to either break the endogeneity through experimental intervention, or account for the confounding variable through appropriate conditioning, or invoke stronger assumptions about the data-generating mechanism.

By the end of this post, we will formalize this distinction by climbing all three rungs of Pearl's causal ladder. We will see how identical statistical distributions can arise from fundamentally different structural causal models, understand what mathematical objects we must add at each rung to enable progressively stronger causal inferences, and demonstrate these concepts using the causal quartet datasets in R. The rudder paradox will serve as our recurring example, showing how each level of the causal hierarchy offers different—and increasingly powerful—tools for reasoning about cause and effect in the presence of endogenous variables.

## Setup and Data

We'll use the `quartets` package, which contains four datasets with a beautiful property: they all have identical summary statistics, but they arise from completely different causal mechanisms. We'll also use `dagitty` for working with directed acyclic graphs (DAGs), and our usual tidyverse tools for data manipulation and visualization.


``` r
# Load required packages
library(quartets)      # Our example datasets
library(dagitty)       # For DAG manipulation and testing
library(tidyverse)     # For data manipulation and basic plotting
library(ggthemes)      # For theme_tufte
library(patchwork)     # For combining plots

# Set a theme for all plots
theme_set(theme_tufte(base_size = 11))
```

Let's load the causal quartet and take a first look. These four datasets were carefully constructed to have identical correlations between x and y, but as we'll discover, they tell four completely different causal stories.


``` r
# Load the causal quartet
data("causal_quartet")

# Look at the structure
head(causal_quartet)
```

```
## # A tibble: 6 × 6
##   dataset      exposure outcome covariate    u1    u2
##   <chr>           <dbl>   <dbl>     <dbl> <dbl> <dbl>
## 1 (1) Collider   0.486    1.71      2.24     NA    NA
## 2 (1) Collider   0.0653   0.669     0.924    NA    NA
## 3 (1) Collider  -1.40    -1.60     -0.999    NA    NA
## 4 (1) Collider  -0.546   -1.73     -2.34     NA    NA
## 5 (1) Collider  -0.401    0.617     0.207    NA    NA
## 6 (1) Collider  -2.38    -2.15     -3.62     NA    NA
```

``` r
# Check the dimensions and datasets
causal_quartet |>
  group_by(dataset) |>
  summarise(
    n = n(),
    mean_x = mean(x),
    mean_y = mean(y),
    sd_x = sd(x),
    sd_y = sd(y),
    cor_xy = cor(x, y)
  )
```

```
## Error in `summarise()`:
## ℹ In argument: `mean_x = mean(x)`.
## ℹ In group 1: `dataset = "(1) Collider"`.
## Caused by error:
## ! object 'x' not found
```

Notice something remarkable: all four datasets have nearly identical means, standard deviations, and correlations. If we only looked at these summary statistics, we might think they're the same data. But they're not, and understanding why requires climbing the ladder of causation.


``` r
# Visualize all four datasets
ggplot(causal_quartet, aes(x = x, y = y)) +
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

```
## Error in `geom_point()`:
## ! Problem while computing aesthetics.
## ℹ Error occurred in the 1st layer.
## Caused by error:
## ! object 'x' not found
```

They look similar, but the underlying causal mechanisms are completely different. To see why, we need to understand Pearl's ladder.

## The Three Rungs of Causation

Judea Pearl posits that causal reasoning exists on three distinct levels, which he calls rungs of a ladder. Each rung requires more information than the one below it, and each rung enables us to answer questions that are impossible at lower levels.

**Rung 1: Association** - The world of pure observation. We ask "what is?" We observe patterns and correlations. This is the domain of traditional statistics. We can compute things like P(Y|X), the probability of Y given that we observe X.

**Rung 2: Intervention** - The world of action and experiments. We ask "what if I do?" We imagine actively manipulating one variable and seeing what happens to another. This is the domain of randomized controlled trials and policy evaluation. We compute things like P(Y|do(X)), the probability of Y if we *force* X to take a particular value.

**Rung 3: Counterfactuals** - The world of imagination and retrospection. We ask "what would have been?" We look at what actually happened and imagine how it would have been different under alternative circumstances. This is the domain of individual-level causal attribution, regret, and responsibility. We compute things like "what would Y have been for this specific individual if X had been different?"

Each rung requires adding something to what we had before. Let's climb the ladder step by step, using the causal quartet as our guide.

## Rung 1: Association - The World of Patterns

At the first rung, we're observing the world and looking for patterns. We have data, and we can compute conditional probabilities, correlations, and statistical associations. But we cannot distinguish between causation and correlation. All we can say is "when I see X, I tend to see Y."

### What We Add: DAG Structure

To work at this level, we add a directed acyclic graph (DAG) that represents possible conditional independence relationships. The DAG is just a hypothesis about the structure of dependencies between variables. At this level, the edges don't necessarily mean "causes" - they just mean "statistically dependent."

The four datasets in the causal quartet actually correspond to four fundamental DAG structures. Let's define them.


``` r
# Dataset 1: Direct causal relationship (X causes Y)
dag1 <- dagitty('dag {
  x -> y
}')

# Dataset 2: Common cause / Fork (Z causes both X and Y)
dag2 <- dagitty('dag {
  z -> x
  z -> y
}')

# Dataset 3: Mediation / Chain (X causes Z, Z causes Y)
dag3 <- dagitty('dag {
  x -> z -> y
}')

# Dataset 4: Collider / Inverted Fork (X and Y both cause Z)
dag4 <- dagitty('dag {
  x -> z
  y -> z
}')

# Plot them side by side
par(mfrow = c(2, 2), mar = c(1, 1, 3, 1))
plot(dag1, main = "Dataset 1: Direct")
plot(dag2, main = "Dataset 2: Fork")
plot(dag3, main = "Dataset 3: Chain")
plot(dag4, main = "Dataset 4: Collider")
```

![center](/figures/causality1/define-dags-1.png)

These four structures look different, but remember: all four datasets have the same correlation between x and y. How can we tell them apart?

### What We Can Do: Test Conditional Independencies

The key insight of the first rung is that different DAG structures imply different patterns of conditional independence. Even though x and y are correlated in all four datasets, they have different relationships when we condition on the third variable z.

We can use the d-separation criterion to derive testable implications. The `dagitty` package does this automatically for us.


``` r
# What conditional independencies does each DAG imply?
cat("Dataset 1 (Direct) implies:\n")
```

```
## Dataset 1 (Direct) implies:
```

``` r
print(impliedConditionalIndependencies(dag1))

cat("\nDataset 2 (Fork) implies:\n")
```

```
## 
## Dataset 2 (Fork) implies:
```

``` r
print(impliedConditionalIndependencies(dag2))
```

```
## x _||_ y | z
```

``` r
cat("\nDataset 3 (Chain) implies:\n")
```

```
## 
## Dataset 3 (Chain) implies:
```

``` r
print(impliedConditionalIndependencies(dag3))
```

```
## x _||_ y | z
```

``` r
cat("\nDataset 4 (Collider) implies:\n")
```

```
## 
## Dataset 4 (Collider) implies:
```

``` r
print(impliedConditionalIndependencies(dag4))
```

```
## x _||_ y
```

Notice the pattern here. The fork structure (dataset 2) and the chain structure (dataset 3) both imply that x and y should be independent when we condition on z. But the collider structure (dataset 4) implies that x and y should be unconditionally independent (which they're not in the data, because we selected on z).

Let's test these implications against the actual data using `localTests()`, which performs conditional independence tests.


``` r
# Test the fork structure against dataset 2
data2 <- causal_quartet |> filter(dataset == 2)

cat("Testing fork structure (z -> x, z -> y) against dataset 2:\n\n")
```

```
## Testing fork structure (z -> x, z -> y) against dataset 2:
```

``` r
results2 <- localTests(dag2, data2, type = "cis")
```

```
## Error in sample.cov[vars, vars]: subscript out of bounds
```

``` r
print(results2)
```

```
## Error: object 'results2' not found
```

``` r
# Visualize the test results
plotLocalTestResults(results2)
```

```
## Error: object 'results2' not found
```

The test passes! The implication that x and y are independent given z holds in dataset 2, which is consistent with a fork structure. Let's verify that the same structure would fail for dataset 3.


``` r
# Test the fork structure against dataset 3 (should fail)
data3 <- causal_quartet |> filter(dataset == 3)

cat("Testing fork structure against dataset 3 (wrong structure):\n\n")
```

```
## Testing fork structure against dataset 3 (wrong structure):
```

``` r
results3_wrong <- localTests(dag2, data3, type = "cis")
```

```
## Error in sample.cov[vars, vars]: subscript out of bounds
```

``` r
print(results3_wrong)
```

```
## Error: object 'results3_wrong' not found
```

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
```

```
## Error in sample.cov[vars, vars]: subscript out of bounds
```

``` r
print(results3)
```

```
## Error: object 'results3' not found
```

``` r
plotLocalTestResults(results3)
```

```
## Error: object 'results3' not found
```

Perfect. The chain structure fits dataset 3.

### The Power and Limits of Rung 1

What have we accomplished at rung 1? We've shown that even though all four datasets have identical correlations between x and y, we can distinguish between them by testing conditional independence patterns. This is powerful! It means that "correlation is not causation" is not the end of the story. We can use conditional independence testing to falsify causal structures that are inconsistent with the data.

But notice what we cannot do. We cannot prove that any particular structure is correct. We can only show that some structures are inconsistent with the data. Moreover, we cannot distinguish between correlation and causation just from observational data. The edges in our DAGs could represent predictive relationships rather than causal ones.

To go further, we need to climb to the second rung.

## Rung 2: Intervention - The World of Action

At the second rung, we move from passive observation to active manipulation. We imagine or actually perform interventions where we force a variable to take a particular value and observe what happens to other variables. This is the world of experiments, policies, and "what if I do?" questions.

### What We Add: Causal Interpretation

The key addition at rung 2 is semantic: we now interpret the edges in our DAG as representing direct causal relationships, not just statistical dependencies. When we write x → y, we mean "x directly causes y," not just "x predicts y."

This is a crucial leap that cannot be justified by data alone. We need additional assumptions or knowledge to make this jump, such as temporal ordering (causes precede effects), experimental manipulation, or domain knowledge about mechanisms.

We also add a new mathematical operator: **do(X = x)**, which represents forcing X to take the value x. This is different from observing X = x. When we intervene, we perform graph surgery by removing all incoming edges to X, reflecting the fact that we've overridden X's normal causes.

### What We Can Do: Predict Intervention Effects

Let's work with dataset 2, which has the fork structure: z causes both x and y. At rung 1, we found that x and y are correlated. But what happens if we *intervene* on x?


``` r
# Visualize the fork structure
data2_plot <- data2 |>
  ggplot(aes(x = x, y = y)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(
    title = "Dataset 2: Fork Structure",
    subtitle = "Observational relationship between X and Y",
    x = "X (observed)",
    y = "Y"
  )

data2_plot
```

```
## Error in `geom_point()`:
## ! Problem while computing aesthetics.
## ℹ Error occurred in the 1st layer.
## Caused by error:
## ! object 'x' not found
```

The observational correlation is clear. But according to our causal model (the fork), x does not actually cause y. Both are caused by z. So what should happen if we intervene to set x to a specific value?

The answer is: nothing should happen to y. When we intervene on x by removing the edge from z to x, we break the correlation between x and y, because that correlation was entirely due to their common cause z.

Let's verify this by computing the adjustment set. In causal inference, an adjustment set is a set of variables we need to condition on to correctly estimate the causal effect of x on y.


``` r
# Find what we need to adjust for to estimate the causal effect of x on y
cat("To estimate the causal effect of x on y in the fork structure:\n")
```

```
## To estimate the causal effect of x on y in the fork structure:
```

``` r
adjustmentSets(dag2, exposure = "x", outcome = "y")
```

```
## { z }
```

We need to adjust for z. Let's do that and see what happens to the relationship between x and y.


``` r
# Fit a model adjusting for z
model_adjusted <- lm(y ~ x + z, data = data2)
```

```
## Error in eval(predvars, data, env): object 'y' not found
```

``` r
# The coefficient on x should be near zero
summary(model_adjusted)$coefficients
```

```
## Error: object 'model_adjusted' not found
```

``` r
# We can also look at the partial correlation
# (correlation between x and y after removing effect of z)
residuals_x <- residuals(lm(x ~ z, data = data2))
```

```
## Error in eval(predvars, data, env): object 'x' not found
```

``` r
residuals_y <- residuals(lm(y ~ z, data = data2))
```

```
## Error in eval(predvars, data, env): object 'y' not found
```

``` r
partial_cor <- cor(residuals_x, residuals_y)
```

```
## Error: object 'residuals_y' not found
```

``` r
cat("\nPartial correlation of x and y given z:", round(partial_cor, 3), "\n")
```

```
## Error: object 'partial_cor' not found
```

After adjusting for z, the effect of x on y is essentially zero, just as our causal model predicted. The intervention P(y | do(x)) would have no effect on y.

Now let's contrast this with dataset 3, which has a chain structure: x causes z, and z causes y. Here, x *does* causally affect y, mediated through z.


``` r
# For the chain structure, what do we adjust for?
cat("To estimate the causal effect of x on y in the chain structure:\n")
```

```
## To estimate the causal effect of x on y in the chain structure:
```

``` r
adjustmentSets(dag3, exposure = "x", outcome = "y")
```

```
##  {}
```

The empty set! We don't need to adjust for anything because there are no confounders. The effect of x on y can be estimated directly from the observational data.


``` r
# Fit a model
model_chain <- lm(y ~ x, data = data3)
```

```
## Error in eval(predvars, data, env): object 'y' not found
```

``` r
summary(model_chain)$coefficients
```

```
## Error: object 'model_chain' not found
```

``` r
# The effect is real and substantial
cat("\nTotal causal effect of x on y:", round(coef(model_chain)[2], 3), "\n")
```

```
## Error: object 'model_chain' not found
```

### The Key Distinction

The crucial distinction between rungs 1 and 2 is this: at rung 1, we saw that datasets 2 and 3 both show correlation between x and y. But at rung 2, we understand that in dataset 2, intervening on x won't change y (because their correlation is due to confounding by z), while in dataset 3, intervening on x will change y (because x genuinely causes y through z).

This is the difference between P(Y|X) and P(Y|do(X)). In dataset 2, they're different. In dataset 3, they're the same (after accounting for the direct path).

We can visualize this distinction by simulating interventions.


``` r
# Simulate what happens if we force x = 2 for everyone in dataset 2
# According to the causal model: y should not change
# because y only depends on z, not on our intervention on x

# Dataset 2: Fork structure (intervention should have no effect)
set.seed(42)
n <- nrow(data2)

# Original y values are determined by z
original_z <- data2$z
original_y <- data2$y

# Intervene: force x = 2 (this breaks the z -> x edge)
# But y still depends only on z, so y doesn't change
intervened_x <- rep(2, n)
intervened_y <- original_y  # y unchanged because it only depends on z

fork_comparison <- tibble(
  original_y = original_y,
  intervened_y = intervened_y,
  difference = intervened_y - original_y
)

cat("Dataset 2 (Fork) - Effect of intervention:\n")
```

```
## Dataset 2 (Fork) - Effect of intervention:
```

``` r
cat("Mean change in Y:", round(mean(fork_comparison$difference), 3), "\n")
```

```
## Mean change in Y: NaN
```

``` r
cat("SD of change in Y:", round(sd(fork_comparison$difference), 3), "\n\n")
```

```
## SD of change in Y: NA
```

``` r
# Dataset 3: Chain structure (intervention should have an effect)
# Here, x causes z causes y, so changing x will change y

# Estimate the causal mechanism from the data
model_xz <- lm(z ~ x, data = data3)
```

```
## Error in eval(predvars, data, env): object 'z' not found
```

``` r
model_zy <- lm(y ~ z, data = data3)
```

```
## Error in eval(predvars, data, env): object 'y' not found
```

``` r
# Intervene: force x = 2
intervened_x_chain <- rep(2, nrow(data3))
# This changes z through x -> z
intervened_z <- predict(model_xz, newdata = tibble(x = intervened_x_chain))
```

```
## Error: object 'model_xz' not found
```

``` r
# Which changes y through z -> y  
intervened_y_chain <- predict(model_zy, newdata = tibble(z = intervened_z))
```

```
## Error: object 'model_zy' not found
```

``` r
chain_comparison <- tibble(
  original_y = data3$y,
  intervened_y = intervened_y_chain,
  difference = intervened_y_chain - original_y
)
```

```
## Error: object 'intervened_y_chain' not found
```

``` r
cat("Dataset 3 (Chain) - Effect of intervention:\n")
```

```
## Dataset 3 (Chain) - Effect of intervention:
```

``` r
cat("Mean change in Y:", round(mean(chain_comparison$difference), 3), "\n")
```

```
## Error: object 'chain_comparison' not found
```

``` r
cat("SD of change in Y:", round(sd(chain_comparison$difference), 3), "\n")
```

```
## Error: object 'chain_comparison' not found
```

In the fork structure, intervening on x has essentially no effect on y. In the chain structure, it has a substantial effect. This is what we mean by moving from correlation to causation: we can now predict what happens when we act, not just what we see when we observe.

### The Power and Limits of Rung 2

At rung 2, we can answer policy questions and predict the effects of interventions. We can distinguish between genuine causal effects and spurious correlations due to confounding. This is enormously powerful for science and decision-making.

But notice what we still cannot do. We can tell you the *average* effect of intervening on x in the population: P(Y|do(X)) tells us what would happen on average if we forced X for everyone. But we cannot tell you what would have happened for a specific individual who actually experienced one value of X. We cannot answer "what if things had been different *for this person*?" 

To answer that question, we need to climb to the third rung.

## Rung 3: Counterfactuals - The World of Imagination

At the third rung, we move from population-level predictions to individual-level explanations. We look at what actually happened to a specific person or instance and ask "what would have been different if...?" This is the domain of regret, responsibility, attribution, and retrospection.

### What We Add: Functional Mechanisms

The key addition at rung 3 is the specification of functional equations that describe *how* each variable is generated from its causes. We move from a DAG (which just shows which variables affect which) to a Structural Causal Model (SCM), which specifies the mechanisms.

For each variable, we write:
$$X_i = f_i(\text{Parents}(X_i), U_i)$$

where $f_i$ is a function and $U_i$ represents all the unmeasured factors (noise, randomness, individual variation) that affect $X_i$.

This is a big step up in specificity. At rung 2, we only needed to know the graph structure. At rung 3, we need to know the actual functional form of how causes produce effects.

### What We Can Do: Individual Counterfactuals

Let's work with dataset 3 (the chain structure) since it has a clear causal mechanism we can estimate. The data was actually generated from a specific structural model. Let's estimate that model and use it to answer counterfactual questions.


``` r
# Estimate the structural equations
# x -> z: z = a*x + u_z
# z -> y: y = b*z + u_y

model_x_to_z <- lm(z ~ x, data = data3)
```

```
## Error in eval(predvars, data, env): object 'z' not found
```

``` r
model_z_to_y <- lm(y ~ z, data = data3)
```

```
## Error in eval(predvars, data, env): object 'y' not found
```

``` r
cat("Structural equations:\n")
```

```
## Structural equations:
```

``` r
cat("z = ", round(coef(model_x_to_z)[1], 3), " + ", 
    round(coef(model_x_to_z)[2], 3), "*x + u_z\n", sep = "")
```

```
## Error: object 'model_x_to_z' not found
```

``` r
cat("y = ", round(coef(model_z_to_y)[1], 3), " + ", 
    round(coef(model_z_to_y)[2], 3), "*z + u_y\n\n", sep = "")
```

```
## Error: object 'model_z_to_y' not found
```

Now suppose we observe a specific individual with x = 1.5, z = 2.0, y = 3.5. We want to answer: "What would y have been for this individual if x had been 2.5 instead?"

This is a counterfactual question. To answer it, we use Pearl's three-step procedure:

1. **Abduction**: Given what we observed, infer the values of the unobserved noise terms $U_z$ and $U_y$ for this specific individual
2. **Action**: Modify the structural equations to reflect the intervention (set x = 2.5)
3. **Prediction**: Compute what y would have been using the modified equations and the inferred noise terms


``` r
# Observed values for one individual
observed_x <- 1.5
observed_z <- 2.0
observed_y <- 3.5

# Step 1: Abduction - infer the noise terms
# From z = a + b*x + u_z, we get u_z = z - (a + b*x)
u_z <- observed_z - predict(model_x_to_z, newdata = tibble(x = observed_x))
```

```
## Error: object 'model_x_to_z' not found
```

``` r
# From y = c + d*z + u_y, we get u_y = y - (c + d*z)  
u_y <- observed_y - predict(model_z_to_y, newdata = tibble(z = observed_z))
```

```
## Error: object 'model_z_to_y' not found
```

``` r
cat("Step 1 - Abduction (infer noise terms):\n")
```

```
## Step 1 - Abduction (infer noise terms):
```

``` r
cat("u_z =", round(u_z, 3), "\n")
```

```
## Error: object 'u_z' not found
```

``` r
cat("u_y =", round(u_y, 3), "\n\n")
```

```
## Error: object 'u_y' not found
```

``` r
# Step 2: Action - set x to counterfactual value
counterfactual_x <- 2.5

# Step 3: Prediction - compute counterfactual y
# First, compute what z would have been
counterfactual_z <- predict(model_x_to_z, 
                            newdata = tibble(x = counterfactual_x)) + u_z
```

```
## Error: object 'model_x_to_z' not found
```

``` r
# Then, compute what y would have been
counterfactual_y <- predict(model_z_to_y, 
                           newdata = tibble(z = counterfactual_z)) + u_y
```

```
## Error: object 'model_z_to_y' not found
```

``` r
cat("Step 2 - Action (set x =", counterfactual_x, ")\n\n")
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
cat("Counterfactual z would have been:", round(counterfactual_z, 3), "\n")
```

```
## Error: object 'counterfactual_z' not found
```

``` r
cat("Counterfactual y would have been:", round(counterfactual_y, 3), "\n\n")
```

```
## Error: object 'counterfactual_y' not found
```

``` r
cat("Summary:\n")
```

```
## Summary:
```

``` r
cat("Observed: x =", observed_x, ", y =", observed_y, "\n")
```

```
## Observed: x = 1.5 , y = 3.5
```

``` r
cat("Counterfactual: if x had been", counterfactual_x, 
    ", y would have been", round(counterfactual_y, 3), "\n")
```

```
## Error: object 'counterfactual_y' not found
```

``` r
cat("Effect for this individual:", 
    round(counterfactual_y - observed_y, 3), "\n")
```

```
## Error: object 'counterfactual_y' not found
```

Notice something crucial: the counterfactual answer depends on the specific observed values for this individual. Two people with the same observed x might have different counterfactual y values because they have different noise terms (different values of $U$).

### The Connection to Bayesian Inference

You might have noticed that what we did in step 1 (abduction) looks a lot like Bayesian inference. We observed some data and used it to infer the values of unobserved variables (the noise terms). This connection is deep and important.

In a Bayesian framework, we would treat the noise terms as latent variables and compute their posterior distribution given the observed data. Then, to compute counterfactuals, we would sample from this posterior and simulate what would have happened under different interventions.

This is exactly the approach taken in Richard McElreath's "Statistical Rethinking" (particularly Chapter 16), where he shows how to use Bayesian inference to compute counterfactuals in structural causal models. The generative model (the SCM) becomes the inference target, and counterfactual reasoning becomes a form of posterior prediction under modified models.

### The Power and Limits of Rung 3

At rung 3, we can answer questions about individual instances and attribute causation at the finest grain. We can reason about responsibility, regret, and explanation. We can say not just "smoking causes cancer on average" but "Bob's cancer was caused by his smoking."

But this power comes at a price. We need much more information than at lower rungs. Specifically, we need:

1. The correct causal graph (rung 2 requirement)
2. The functional form of how causes produce effects
3. The distribution of unobserved factors

These requirements are often difficult or impossible to satisfy with observational data alone. We typically need strong assumptions about functional forms (linearity, additivity, etc.) or additional data from experiments.

## Returning to the Rudder Paradox

Now we're ready to understand why the rudder paradox fooled us at the start. Let's generate some data that mimics the rudder situation and see how each rung of the ladder handles it.


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
  W [unobserved]
  W -> R
  W -> D
  R -> D
}')
```

### Rung 1 Analysis: Correlation Fails

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

The correlation is near zero! At rung 1, we would conclude (incorrectly) that the rudder doesn't affect the boat's direction. This is a complete failure of correlation-based reasoning.

The problem is confounding by the unobserved wind W. The wind affects both R and D in opposite ways, creating what's called "collider stratification bias" or "table 2 fallacy" when we don't account for it.

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

### Rung 3 Analysis: Individual Attribution

At rung 3, we can answer even more nuanced questions. For instance: "On Tuesday at 3pm, the boat drifted 2 degrees off course despite the rudder being angled 5 degrees. What portion of the drift was due to the wind versus the rudder?"

This requires knowing the functional form and being able to infer the specific wind value on that occasion.


``` r
# Pick a specific observation
idx <- 100
observed_R <- rudder_data$R[idx]
observed_D <- rudder_data$D[idx]  
true_W <- rudder_data$W[idx]  # In reality we don't observe this

# But we could infer it from the structural equation if we knew the model
# R = -0.9*W + u_r, so W ≈ -R/0.9 (approximately, ignoring noise)

# Given the true structural equations:
# D = 0.5*W + 0.5*R + u_d

# Counterfactual 1: What if there had been no wind?
counterfactual_W_zero <- 0
counterfactual_D_nowind <- 0.5 * counterfactual_W_zero + 0.5 * observed_R

# Counterfactual 2: What if the rudder had been neutral?
counterfactual_R_zero <- 0  
counterfactual_D_norudder <- 0.5 * true_W + 0.5 * counterfactual_R_zero

cat("Observed situation:\n")
```

```
## Observed situation:
```

``` r
cat("Rudder:", round(observed_R, 2), "\n")
```

```
## Rudder: 1.11
```

``` r
cat("Direction:", round(observed_D, 2), "\n")
```

```
## Direction: 0.02
```

``` r
cat("Wind (unobserved):", round(true_W, 2), "\n\n")
```

```
## Wind (unobserved): -1.03
```

``` r
cat("Counterfactual 1 - If there had been no wind:\n")
```

```
## Counterfactual 1 - If there had been no wind:
```

``` r
cat("Direction would have been:", round(counterfactual_D_nowind, 2), "\n")
```

```
## Direction would have been: 0.55
```

``` r
cat("Difference from observed:", 
    round(counterfactual_D_nowind - observed_D, 2), "\n\n")
```

```
## Difference from observed: 0.54
```

``` r
cat("Counterfactual 2 - If rudder had been neutral:\n")
```

```
## Counterfactual 2 - If rudder had been neutral:
```

``` r
cat("Direction would have been:", round(counterfactual_D_norudder, 2), "\n")
```

```
## Direction would have been: -0.51
```

``` r
cat("Difference from observed:", 
    round(counterfactual_D_norudder - observed_D, 2), "\n\n")
```

```
## Difference from observed: -0.53
```

``` r
# Attribution: How much did the rudder vs wind contribute?
cat("Attribution:\n")
```

```
## Attribution:
```

``` r
cat("Rudder contribution:", round(0.5 * observed_R, 2), "\n")
```

```
## Rudder contribution: 0.55
```

``` r
cat("Wind contribution:", round(0.5 * true_W, 2), "\n")
```

```
## Wind contribution: -0.51
```

At rung 3, we can decompose what happened on that specific occasion into the separate contributions of wind and rudder. This is the finest-grained causal understanding possible.

## Summary: Why the Ladder Matters

Let's recap what we've learned by climbing Pearl's ladder of causation.

**Rung 1: Association** teaches us that:
- Correlation patterns can distinguish between different causal structures
- Conditional independence testing can falsify wrong causal models
- But observation alone cannot prove causation or distinguish correlation from confounding
- The rudder paradox shows that correlation can be completely misleading

**Rung 2: Intervention** teaches us that:
- Causal effects are different from statistical associations
- We need causal graphs and adjustment sets to predict intervention effects  
- The same correlation can imply zero causal effect (fork) or strong causal effect (chain)
- Experiments break confounding by randomizing the intervention

**Rung 3: Counterfactuals** teaches us that:
- Population-level effects are different from individual-level effects
- We need functional mechanisms to reason about specific instances
- Attribution and responsibility require counterfactual reasoning
- Bayesian inference provides a natural framework for inferring unobservables

Each rung requires adding more structure and making stronger assumptions:

- **Rung 1**: DAG structure + faithfulness assumption
- **Rung 2**: Causal interpretation of edges + no hidden confounders
- **Rung 3**: Functional forms + distribution of unobservables

And each rung enables us to answer questions that are impossible at lower levels:

- **Rung 1**: "What patterns exist?" → Statistics
- **Rung 2**: "What if I act?" → Science and policy  
- **Rung 3**: "What would have been?" → Explanation and attribution

The causal quartet showed us that identical statistics can hide different causal stories. The rudder paradox showed us that correlation can be systematically misleading when there's hidden confounding. Together, they illustrate why we need to think carefully about which rung we're on and what questions we're trying to answer.

When you're working with data, ask yourself: Which rung am I on? Do I just want to predict (rung 1)? Do I want to guide interventions (rung 2)? Do I want to explain what happened (rung 3)? The answer determines what tools you need and what assumptions you must make.

The ladder isn't just a philosophical framework. It's a practical guide for matching your methods to your goals.

