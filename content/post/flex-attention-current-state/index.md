---
title: Flex Attention as of Sept 2024
subtitle: My experience with hacking around with flex attention 
date: 2024-09-22T05:37:15.089Z
summary: 
draft: true
featured: false
commentable: true
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---

I wanted to quickly jot down some learnings after playing around with the awesome new flex attention API: https://pytorch.org/blog/flexattention/ 
If you want to use it yourself, you should use the latest torch 2.5.0 package ([branch cut details here](https://dev-discuss.pytorch.org/t/pytorch-2-5-release-branch-cut-for-pytorch-core-is-completed/2452)). Flex attention allows you to define custom masks in torch while being as performant as writnig your own custom kernels (conditions apply :p). 
