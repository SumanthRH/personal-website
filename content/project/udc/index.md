---
title: Image Restoration for Under-Display Cameras
summary: Under-display cameras offer a number of benefits, but the images aren't great yet. We trained a deep network for image restoration, published in ECCV 2020. 

tags:
  - ML
  - Computer Vision
date: 2020-08-01
# external_link: 
---
TL; DR: We created a novel deep learning based model for image restoration, resulting in a publication at RLQ-UDC Workshops, ECCV 2020. We also placed 2nd /150 teams at the Under Display Camera Challenge. Check out my presentation [here](https://youtu.be/WnNOg178iSk)!


Under Display Cameras present a promising opportunity for phone manufacturers to achieve bezel-free displays by positioning the camera behind semi-transparent OLED screens. Unfortunately, such imaging systems suffer from severe image degradation due to light attenuation and diffraction effects. Presenting Deep Atrous Guided Filter (DAGF), a two-stage, end-to-end approach for image restoration in UDC systems. A Low-Resolution Network first restores image quality at low-resolution, which is subsequently used by the Guided Filter Network as a filtering input to produce a high-resolution output. Besides the initial downsampling, our low-resolution network uses multiple, parallel atrous convolutions to preserve spatial resolution and emulates multi-scale processing. Our approach's ability to directly train on megapixel images results in significant performance improvement. We additionally propose a simple simulation scheme to pre-train our model and boost performance. Our overall framework ranks 2nd and 5th in the RLQ-TOD'20 UDC Challenge for POLED and TOLED displays, respectively. More details on the [project page](https://varun19299.github.io/deep-atrous-guided-filter/).
