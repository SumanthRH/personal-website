---
title: Low-resource Satellite Image Segmentation
summary: Achieved pixel-wise segmentation for satellite images with a dataset of just 14 images!


tags:
  - ML
date: 2018-12-01
---

This project was our submission to a challenge in [Inter IIT Tech Meet 2018](https://www.iitb.ac.in/en/event/7th-inter-iit-tech-meet),
an annual competition held among the 20 odd IITs in India. With a dataset of just 14 images, we were tasked with segmenting
pixels in satellite images into 8 classes - Roads, Buildings, Trees, Grass, Bare Soil, Water, Railways and Swimming
pools. The images were multispectral - we had access to a Near InfraRed (NIR) channel along with the usual RGB. Our approach
was to handle the task class wise and use a mix of classical computer vision and deep learning. For classes such as 
Grass, one can in fact use the NIR channel to get an almost pixel-wise accurate segmentation mask without any machine learning!
More details can be found in the report [here](https://github.com/iitmcvg/eye-in-the-sky/blob/master/InterIIT_2018-IITM.pdf) and our
code is available [here](https://github.com/iitmcvg/eye-in-the-sky). 
