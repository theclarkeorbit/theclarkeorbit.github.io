---
title: "Linear and mixed integer programming"
Date: 08 Sep 2018
---

post inspired by the OptiPy meetup. [link to meetup](https://www.meetup.com/OptiPy-Python-Quants-of-BeNeDeLux/events/253090625/)

## Linear programming
[Wiki article on linear programming](https://en.wikipedia.org/wiki/Linear_programming). Combining the concept of an objective function with constraints. Dantzig came up with the [simplex algorithm](https://en.wikipedia.org/wiki/Simplex_algorithm). The most common solver in python is [PuLP](https://pythonhosted.org/PuLP/).

Optimisation problems where the constraints and cost function are linear, and the decision variables are continuous, are the simple, canonical domain of linear programming. However, such problems can be solved in polynomial time, which means that to tackle a hard (NP hard) problem with this framework, the problem definition needs to be exponentially large ([see this (pdf)](https://www.cwi.nl/system/files/scimeeting13.pdf)).

However, if we allow for (some) decision variables to be integers instead of reals, a much richer range of problems can be expressed and solved in a similar framework. This is called mixed integer programming.

First, a toy linear programming problem in PuLP.

### Simple linear programming

Eat the optimal amount of schnitzel and pommes to survive, given nutrition value and costs in terms of various ingredients. Assume that schnitzel and pommes come in continuous quantities, and humans need at least 150 units of carbs and 50 units of proteins to survive (schnitzel has 23 units of carbs and 18 units of proteins, while pommes has 33 units of carbs and 4 units of proteins) while consuming less than 75 units of fat (schnitzel has 15 units of fat and pommes 13).

This problem can be formulated as follows :

Decision vars :
- how much schnitzel : x1
- how much pommes : x2

```
cost = 8*x1 + 3*x2 # cost function
```

constraints :
```
 23*x1 + 33*x2 >= 150 # carbs
 18*x1 + 4*x2 >= 50 # protein
 15*x1 + 13*x2 <= 75 # fats
```

we now code this up in the popular python LP framework, PuLP :

```python
import pulp

diet_program = pulp.LpProblem("Diet Program", pulp.LpMinimize)
x1 = pulp.LpVariable("schnitzel", lowBound=0, cat="Continuous")
x2 = pulp.LpVariable("pommes", lowBound=0, cat="Continuous")

diet_program += 8*x1 + 3*x2, "cost"

diet_program += 23*x1 + 33*x2 >= 150
diet_program += 18*x1 + 4*x2 >= 50
diet_program += 15*x1 + 13*x2 <= 75

diet_program.solve()

diet_program.solve()
pulp.LpStatus[diet_program.status]

for variable in diet_program.variables():
    print("{} = {}".format(variable.name, variable.varValue))
```

giving the result

```
pommes = 3.0876494
schnitzel = 2.0916335
```

Often, real problems have a large number of decision variables and constraints, but LP problems remain tractable (by and large) even in high dimensions.

## Mixed integer programming

Mixed integer models, however, are a different story. Large models are very hard to solve with open source solvers and expensive commercial solvers are needed to solve real world problems in a reasonable amount of time.

### Material

1. the Sagemath documentation page has a rather [good chapter](http://doc.sagemath.org/html/en/thematic_tutorials/linear_programming.html) on linear and mixed integer programming. On the face of it, the syntax seems more elegant and extensible than [PuLP](https://pythonhosted.org/PuLP/)
2. For R, most optimisation problems need matrix definitions and this makes constructing large models in R basically impossible. The [ompr package](https://channel9.msdn.com/Events/useR-international-R-User-conferences/useR-International-R-User-2017-Conference/ompr-an-alternative-way-to-model-mixed-integer-linear-programs) seems to solve this issue, and enables construction of models in a step by step fashion, like PuLP and sagemath.

Below, we will work through a couple of relatively simple problems in sagemath, R and PuLP.

### Tutorial - knapsack problem.

#### Sagemath version

see [this page](http://doc.sagemath.org/html/en/thematic_tutorials/linear_programming.html) for an intro and sage code.

We have some objects L each with some weights and some usefulness. We can carry a maximum weight of C, while optimising the total usefulness of the objects we pack.

Below, we assign random weights and usefulness to our objects-

```python
C = 1
L = ["pan", "book", "knife", "gourd", "flashlight"]
L.extend(["random_stuff_" + str(i) for i in range(20)])
weight = {}
usefulness = {}
set_random_seed(685474)
for o in L:
    weight[o] = random()
    usefulness[o] = random()
```

We now define the mixed integer model.
The objective to be maximised is usefulness of taken objects, the constraint is the maximum weight C and the only decision variables are an array of binary variables corresponding to each objects, determining if they are taken or not.

```python
p = MixedIntegerLinearProgram()
taken = p.new_variable(binary=True)
p.add_constraint(sum(weight[o] * taken[o] for o in L) <= C)
p.set_objective(sum(usefulness[o] * taken[o] for o in L))
```

Having set up the model, we solve it using the in-built optimizer.

```python
p.solve() # abs tol 1e-6
taken = p.get_values(taken)
print('the total weight of taken objects is')
sum(weight[o] * taken[o] for o in L)
```

which gives the expected result.
```
the total weight of taken objects is
0.6964959796619171
```

#### R version

Most optimisers in R (including the popular [ROI](https://cran.r-project.org/web/packages/ROI/) package that provides a unified interface to multiple solvers) need problem definitions in matrix form. However, matrix definitions are not easy to read and interpret, and are the end result of a process of problem definition. Step by step definitions of the optimisation problem (like the one above) provide a much better framework in which to understand and define optimisation problems.

The [ompr](https://dirkschumacher.github.io/ompr/) package provides such an interface for problem definition and solution in R, using the pipe operator from the [Tidyverse](https://www.tidyverse.org/).

Setting up the basics

```r
library(tidyverse)
library(ROI)
library(ROI.plugin.glpk)
library(ompr)
library(ompr.roi)

C <- 1 # max weight
L <- list("torch", "food", "tent", "knife", "books") # objects
for(j in 1:20) {
    L <- append(L, paste('random_stuff',j,sep = '_'))
}
weight_list <- list()
usefulness_list <- list()
set.seed(685474)
for(j in 1:length(L)) {
    weight_list <- append(weight_list, unlist(runif(1)))
    usefulness_list <- append(usefulness_list, unlist(runif(1)))
}
usefulness <- as.numeric(usefulness_list)
weight <- as.numeric(weight_list)
```

Defining the model

```r
n = length(L)
model_mip <- MIPModel() %>%
    add_variable(x[i], i=1:n, type = "binary") %>%
    set_objective(sum_expr(usefulness[i]*x[i], i=1:n), "max") %>%
    add_constraint(sum_expr(weight[i]*x[i], i=1:n) <= C)
```

Solving the model with the `glpk`  solver.

```r
solution <- model_mip %>%
    solve_model(with_ROI(solver = "glpk")) %>%
    get_solution(x[i]) %>%
    arrange(i)
```

And then checking what the weight of our knapsack is !

```r
solution_data <- data_frame(objects = unlist(L), usefulness = usefulness, weights = weight, taken = solution$value)
objects_taken <- solution_data %>%
    filter(taken>0.5)
print(sum(objects_taken$weights))
```

and it is

```
[1] 0.949882
```

#### PuLP version

The pulp problem is setup in a very similar fashion to the sage problem.  Below is the entire code for the problem solution in PuLP.

```python
import pulp
import random
import pandas as pd

C = 1.0 # the max weight
L = ["torch", "food", "tent", "knife", "books"]
L.extend(["random_stuff" + str(i) for i in range(20)])
weight = {}
usefulness = {}
random.seed(685474)
for o in L:
    weight[o] = random.uniform(0,1)
    usefulness[o] = random.uniform(0,1)

#the decision variables
x = pulp.LpVariable.dicts('', L, lowBound = 0, upBound = 1, cat = pulp.LpInteger)

# declaring the PuLP model
knapsack_model = pulp.LpProblem("knapsack", pulp.LpMaximize)
# cost function :
knapsack_model += sum([usefulness[thing]*x[thing] for thing in L]), "usefulness"
# constraints
knapsack_model += sum([weight[thing]*x[thing] for thing in L]) <= C

# solving the model
knapsack_model.solve()
pulp.LpStatus[knapsack_model.status]

# displaying the solution in a useful form
total_weight = 0.0
things = {}
for variable in knapsack_model.variables():
    var = variable.name[1:]
    things[var] = variable.varValue
    total_weight += weight[var]*things[var]
    # print("{} = {}".format(var, variable.varValue))

print("the total weight taken is "+str(total_weight))
solution_data = pd.DataFrame([weight, usefulness, things]).T
solution_data.columns = ['weight', 'usefulness', 'taken']
print(solution_data.head())
```

with the output

```
the total weight taken is 0.911966521327
                 weight  usefulness  taken
books          0.977869    0.290585    0.0
food           0.112391    0.445158    1.0
knife          0.735062    0.919826    0.0
random_stuff0  0.881662    0.800397    0.0
random_stuff1  0.888736    0.636453    0.0
```

With these basics out of the way, we can tackle a non-trivial model inspired by real data in the next post.
