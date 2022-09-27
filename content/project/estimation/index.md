---
title: Defending against adversarial attacks through model uncertainty 
summary: Studied different measures and methods of quantifying uncertainty in deep neural nets. I got learn about reasoning with uncertainty and relate that to adversarial attacks!
tags:
  - ML
date: 2020-08-01
---

In this project, we wanted to understand how to build uncertainty aware neural networks - say for classifying an image as a cat/dog, think of this as giving a confidence interval along with a probability estimate. 
We started out in a theoretical setting: we looked at how dropout can be used as a bayesian approximation ([source](http://proceedings.mlr.press/v48/gal16.pdf)). 
The application in mind was adversarial attacks - using uncertainty estimates, we can defend against a malicious agent trying to trick the model. 
We studied different measures of uncertainty used, the pros and cons as they relate to our application. Finally, we
analysed recent work on using Evidence Theory ([source](https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty.pdf))
 to get these uncertainty estimates. Turns out, this is much better against adversarial attacks. Please check the [project repo](https://github.com/SumanthRH/EE5111_Estimation_Theory/tree/master/project) for more details!
