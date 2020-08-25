---
title: "Becoming a Data Scientist : an opinionated take in 2020"
date: 25 Aug 2020
output:
  html_document:
    df_print: paged
---

There are many articles on this subject, and many of them are excellent. For some examples see [here](https://towardsdatascience.com/how-to-become-a-data-scientist-2a02ed565336), [here](https://towardsdatascience.com/how-to-become-a-data-scientist-3f8d6e75482f), [here](https://www.kdnuggets.com/2018/05/simplilearn-9-must-have-skills-data-scientist.html) and [here](https://www.discoverdatascience.org/career-information/data-scientist/). All such articles are opinionated takes, and helped me out to some extent when I was starting out in this field some years ago. Contrary to my expectations, the term **Data Scientist** has become more confusing with time over the last 3-4 years.

This is my poor attempt to clarify (to myself) what I (and a lot of other people) mean when I use this term. This also helps contextualize other terms like "ML engineer" and "Data engineer". 

### A definition, and the basic requirements

**A data scientist uses quantitative techniques to understand, define and solve business problems.** That's it. In this view, a data scientist is a scientist an analytical mind and mathematical techniques to business problems. 

While that works as a definition, it is rather inadequate because it tells us very little about the skills needed and the type of work a data scientist has to do. The answers to those questions depend partly on the context and culture of the business in which a data scientist works, but some skills are likely to be universally required for any data scientist working in the wilderness of corporate life. IMHO, the following skills are the minimum for any data scientist :

- **A bouquet of quantitative skills** : depending on your background, you'll have some quant skills. Monte carlo simulations, probability theory, fourier transforms, etc. The more the better. Apart from those, its good to brush up on statistics, *really* understand linear regression and PCA and get a solid grounding in Bayesian thinking. You could do a lot worse than investing 20-24 hrs in [Richard McElreath's course](https://www.youtube.com/watch?v=4WVelCswXo4&list=PLDcUM9US4XdNM4Edgs7weiyIguLSToZRI) on statistics, bayesian thinking etc. 
- **SQL** : it is hard to imagine a data scientist who does not - very regularly - need to access databases, and for good or ill, SQL is the lingua franca, along with some dialects. Learning SQL well is also a way to learn the paradigms of data wrangling that come in handy later. There are plenty of resources out there to learn SQL, but I quite liked this [free course from Udacity](https://www.udacity.com/course/sql-for-data-analysis--ud198#).  
- **Excel** : every single business uses Excel, and certainly a lot of your stakeholders will use it, and sooner rather than later, you will have to deal with data in excel sheets. Its ubiquity (despite its multifarious flaws) makes it worth learning. I haven't made myself learn it yet, so I can't point to any good resources.  
- **Programming** : It does not really matter what you learn to code in, as long as you can code. The aim should be to feel "I don't care what language or package code I need to deal with, I'll pick it up". But if you must pick one language, I'd recommend Python. Even though I personally prefer R. Learn to write clean code, document it, write tests and version control. [This](https://docs.python-guide.org/intro/learning/) is a great list of free resources.   
- **Data wrangling and visualization** : This is another skill no data scientist can do without. There is no better resource to learn how to approach data once you get hold of it than the [free R4DS book](https://r4ds.had.co.nz/). Yes it is in R, but that is a language worth picking up anyway.   
- **A sprinkling of ML** : Its good to know the classical ML techniques and where you might use them, an excellent, self contained and concise resource is [ISLR](http://faculty.marshall.usc.edu/gareth-james/ISL/) also available freely.   
- **Pursuation and the ability to say NO** : Every data scientist works with multiple stakeholders (often non technical ones) from whome they need to learn, and whome they need to teach. This is unavoidable, and the better you are at listening carefully and communicating well, the happier you will be.   

With these things in your kitty, you are well set to be the problem solver of last resort (and first preference) for your employer. A Sherlock Holmes for business, if you will. However, depending on your context you will probably need some other skills. 

### Context : the dev teams

Increasingly, data scientists are embedded in the engineering departments of their companies, and so (unfortunately) have to jump through such utterly pointless hoops as 5 technical interviews and live coding tests. After you have performed for the engineering manager and convinced 10 developers that you are worth hiring, you will need to be able to interact with their code (this is the easy part) and their organizational and social mores (this is the hard part). Here is what you will need :

- **The phoenix project** : A [book like no other](https://www.amazon.com/Phoenix-Project-DevOps-Helping-Business/dp/0988262592) to understand why things are the way they are in modern software organizations. Added bonus, most devs and product people havent read it so you can quote scripture at them from Day 1.   
- **DevOps** : An understanding of CI/CD pipelines that take your code (the DNA of the thing you are building) and turn it into a running product that does things (the working cell with proteins and interfaces to other cells). LinkedIn has a course called [DevOps for Data Scientists](https://www.linkedin.com/learning/devops-for-data-scientists). Havent watched it, but it'll probably be useful.   
- **Docker + APIs** : Continuing the DNA/Cell analogy, think of a Docker container as the cell wall that encloses the environment in which your code runs. That makes no sense ? It will, eventually. This seems like a [fairly useful introduction](https://www.analyticsvidhya.com/blog/2017/11/reproducible-data-science-docker-for-data-science/). An API is just the interface to the outside world (the stuff the other devs are building) through the cell wall. Take a look at [this](https://www.restapitutorial.com/).  
- **Enough deep learning to tell people why you aren't using deep learning** : this is a cultural issue. Every week, 2-3 devs will ask you why you aren't doing what ever you are doing with deep learning.   


### Context : the marketing/finance departments

Many data scientists are concerned with answering business questions and helping design strategy or informing management decisions. In such situations, the data scientist will probably work closely with a data base team and communicate results in meetings quite a bit. What you will need to succeed :

- **PowerPoint Ninja** : If you can make impactful slides quickly, you are golden. It is worth investing in learning and mastering this artform, for that is what it is. 
- **DWH skills** : Its probably worth being able to lend a hand in maintaining and managing the business data warehouse (and learning what that is, in the first place). You will also want to be familiar with such things as cubes, business KPIs, accounting standards and pivot tables.   


### Context : replacing human cognition  

A lot of companies have realized that one way of utilize data is to make things easier for humans and/or replacing humans for some tasks altogether. These topics have traditionally been the domain of electrical engineering (image and audio processing) and computational linguistics (NLP). If you are expected to work on these fascinating problems, you will need a bouquet of skills that we have not really mentioned before : 

- **Software engineering** : Delivering solutions to these problems - for now - require the data scientist to be more aquainted with robust software engineering techniques than some of the other contexts we have mentioned before. It is probably wise to become very good at writing idiomatic python code, just for starters.   
- **Deep learning** : While these were disparate fields once upon a time, now they are all three (computer vision, NLP, audio processing) sub-fields of "Deep learning" and you will have to master a fairly large amount of material that is common to them all, and quite a lot of (much more interesting) material sepcific to each domain. I unhesitatingly reccommend the [material provided for free by Fast AI](https://www.fast.ai/). If you put in the time and effort to work through it, you will be in an excellent position to contribute. 


### What about those other things ?

An **ML engineer** is a software engineer who trains, deploys and maintains a machine learning model in production. This is something a data scientist might do as well, but not necessarily.  
A **Data engineer** is a software and infrastructure engineer who builds and maintains the cloud infrastructure for the company databases **and** ensures data quality and availability for those in the company who need it, like people who make operational decisions, dashboards, data scientists. This is a hard hard job, and in high demand these days. A data scientist should know some data engineering, if only to appreciate and help the data engineering teams they work with. 

### Take home message  

While there there is an infinite variety of skills one could learn (I have not even touched upon such useful things as Spark and Airflow) I will return to what characterises a data scientist from all the job descriptions around data. **A data scientist uses quantitative techniques to understand, define and solve business problems.** 

A data scientist must have an analytical mind to bring to the table, and must take the effort to develop a deep understanding of the domain and the business context they operate in. Then, they must do whatever it takes (software engineering, organizational lobbying, operational changes) to create impact from the solutions they have devised. 

