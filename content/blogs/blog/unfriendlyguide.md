Title: An unfriendly introduction to causal inference
Date: 2026-05-01

I wrote a book on causal inference. It is short (under 80 pages), it
is called *An Unfriendly Introduction to Causal Inference*, and the
hardcover is now [live on Amazon](https://www.amazon.com/dp/B0GYTWH3FF).

### How I got here

I first read Pearl's *Causality* in 2016 after spotting it on [Peter Jung's](https://www.linkedin.com/in/peterjungx/) bookcase in our office at [KI Analytics](https://bayes.de) in Cologne. It changed how I thought about data, permanently. There is a moment, somewhere in the first chapter, when you realise that the equality sign in a regression has been doing two different jobs all along, and that keeping them straight is not trivial. I waved my copy of [Pearl's *Primer*](https://www.wiley.com/en-us/Causal+Inference+in+Statistics%3A+A+Primer-p-9781119186847) at [Christian Mathissen](https://www.linkedin.com/in/christian-mathissen-06b9a9a3/) when he interviewed me for car2go in 2018, but I was not yet doing causal inference for a living, and the subject stayed in the background.

It came back to the foreground in 2024, when [Navneet Lekshminarayanan](https://www.linkedin.com/in/navneetl/) at
[Holocene](https://holocene.eu) handed me a problem that genuinely demanded causal exploration rather than predictive modelling with a causal story stapled on afterwards. From there came two talks at
Hasgeek's Fifth Elephant — first in [Bangalore](https://hasgeek.com/fifthelephant/2025-winter/sub/your-causal-parrot-might-be-lying-to-you-FhpB6kWkM4AAkYqdYQSCpJ), then in [Pune](https://hasgeek.com/fifthelephant/2026-pune/sub/interrogating-your-twin-causal-reasoning-in-manufa-J2xUQXKPooa7y9jf56bWiH) — and a small series of posts on this blog ([the ladder](climbing-pearls-ladder-of-causation.html), [a workflow on marketing data](a-causal-workflow-in-r-with-coupon-marketing-data.html), [a factory walkthrough](interrogating-your-twin-a-causal-inference-walkthrough.html)).

Preparing the talks and the posts forced me to keep deciding what was
and was not on the critical path. The book is what I had at the end of
that thought process.

### Why another book

There are wonderful books on causal inference. [*The Book of Why*](https://www.basicbooks.com/titles/judea-pearl/the-book-of-why/9780465097616/) is a pleasure. The [*Primer*](https://www.wiley.com/en-us/Causal+Inference+in+Statistics%3A+A+Primer-p-9781119186847) is the shortest textbook entry I know of, and Pearl's
[*Causality*](https://www.cambridge.org/core/books/causality/B0046844FAE10CBF274D4ACBDAEB5F5B) is the foundational text behind most of what we now call the graphical school. From the potential-outcomes side, [Hernán and Robins'
*What If*](https://miguelhernan.org/whatifbook) is generous and free. From the econometric side, Cunningham's [*Mixtape*](https://mixtape.scunning.com/) is excellent and also free. McElreath's [*Statistical Rethinking*](https://xcelab.net/rm/) is the cleanest integration I know of DAG-based thinking with Bayesian modelling.

You should dip into all of them.

The thing is, almost every introduction is written from inside a particular field — epidemiology, economics, or one of the social sciences — and the subject reaches the reader filtered through that field's conventions, its examples, its specific anxieties. If you came from a different quantitative field, you spend a lot of time on the idiosyncrasies of the field the author was writing from before you got to the structural backbone that all of causal inference shares.

So I wrote the version I wanted to read.

### What is in it

The whole book traces a single thread: **DAG → Test → Identify → Estimate**. Every method in it is an elaboration of one of those four steps.

- **Chapter 1** sets up why prediction and causation are different objects, and why a model that predicts beautifully can recommend an intervention that backfires.
- **Chapter 2** develops Pearl's ladder of causation — association, intervention, counterfactual — and the DAG as the language for stating causal assumptions precisely enough that they can be tested and argued with.
- **Chapter 3** ties the ladder to practice through the four-step workflow, on a single running example (a factory floor, machines, shifts, breakdowns) that carries through the rest of the book.
- **Chapters 4 and 5** develop identification and estimation in depth. The first asks whether the question can be answered from observational data at all. The second turns the answer into a number, with classical estimators (outcome regression, IPW, doubly-robust) and a section on sensitivity to unmeasured confounding.
- **Chapter 6** brings in modern machine learning — DML, causal forests, BART, policy learning — and makes the point that causal ML changes the estimator, not the identification.

R packages for every method are pointed to throughout. A Python appendix is on its way and I'll put it up online as a cheatsheet soon.

### Who it is for

If your work involves making sense of data generated by the real world out there — data that arrived from operations, customers, sensors, surveys, transactions — and decisions get made off the back of your analysis, the book is written for you. The structural problem is the same regardless of where the data came from. You have correlations and you need causal claims, and most of the methods you reach for by default do not suffice.

The book is unfriendly only in the sense that it does not hold your hand. It assumes you are willing to become comfortable with ideas concisely expressed in possibly new notation, and it moves quickly because your time is better spent learning the concepts than being eased into them. It assumes you have the usual quantitative toolkit (probability, some linear algebra, the habit of reading equations rather than skipping them) and nothing more.

[Hardcover on Amazon](https://www.amazon.com/dp/B0GYTWH3FF). If you read it and have thoughts — particularly the kind that begin "you should have included..." — I would like to hear them.