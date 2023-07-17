---
title: Integrating $(IA)^3$ with HuggingFace's PEFT 
summary: Implemented and benchmarked $(IA)^3$, a new parameter-efficient fine tuning method. Our code is now a part of HuggingFace's PEFT !

tags:
  - ML
  - NLP
date: 2023-05-01
---

This was a simple course project for the CSE 256: Natural Language Processing course, Spring 2023 at UCSD. Parameter-efficient fine tuning (PEFT) methods are all the rage now, given that they enable you to [finetune large models like Falcon-7B on just Google Colab](https://huggingface.co/blog/falcon)! A new state-of-the-art method called $(IA)^3$ ([Infused Adapter by Inhibiting and Amplifying Inner Activations](https://arxiv.org/abs/2205.05638)) was proposed recently, shown to beat [LoRA](https://arxiv.org/abs/2106.09685)  (the most popular, flexible, and powerful PEFT method) in certain parameter settings! The [official implementation](https://github.com/r-three/t-few) worked only for certain T5-based architectures. Further, the original authors only benchmarked performance on T0. We implemented (and benchmarked) $(IA)^3$ to support encoder-decoder and decoder-only models, and also enabled features like int-8 training, etc. Our implementation is now the official implementatoin for $(IA)^3$ in HuggingFace's PEFT library! [Pull Request](https://github.com/huggingface/peft/pull/578).
