---
title: Everything about Tokenization
summary: A mini-course on the various aspects of LLM tokenization


tags:
  - ML
  - NLP
date: 2023-11-27
---

This was a fun project I worked on in November 2023. Tokenization is an oft-neglected topic in natural language processing. With the recent blow-up of interest in language models, I thought it might be good to step back and really get into the guts of what tokenization is. I created a [repository](https://github.com/SumanthRH/tokenization) to serve as a deep dive into different aspects of tokenization. This was initially supposed to be a single blog, but it evolved into a full mini-course! It's been organized as bite-size chapters for easy navigation, with some code samples and (badly designed) walkthrough notebooks. It is NOT meant to be a complete reference in itself, and is meant accompany other excellent resources like [HuggingFace's NLP course](https://huggingface.co/learn/nlp-course/chapter6/1). The following topics are covered: 

1. [Intro](https://github.com/SumanthRH/tokenization/tree/main/1-intro/): A quick introduction on tokens and the different tokenization algorithms out there. 
2. [BPE](https://github.com/SumanthRH/tokenization/tree/main/2-bpe/): A closer look at the Byte-Pair Encoding tokenization algorithm. We'll also go over a minimal implementation for training a BPE model.
3. [🤗 Tokenizer](https://github.com/SumanthRH/tokenization/tree/main/3-hf-tokenizer/): The internals of HuggingFace tokenizers! We look at state (what's saved by a tokenizer), data structures (how does it store what it saves), and methods (what functionality do you get). We also implement a minimal 🤗 Tokenizer in Python for GPT2.
4. [Challenges with Tokenization](https://github.com/SumanthRH/tokenization/tree/main/4-tokenization-is-hard/): Challenges with integer tokenization, tokenization for non-English languages and going multilingual, with a focus on the recent No Language Left Behind (NLLB) effort from Meta.
5. [Puzzles](https://github.com/SumanthRH/tokenization/tree/main/5-puzzles/): Some simple puzzles to get you thinking about pre-tokenization, vocabulary size, etc.
6. [PostProcessing and more](https://github.com/SumanthRH/tokenization/tree/main/6-postprocessing-and-more/): A look at special tokens and postprocessing, glitch tokens and why you might want to shrink your tokenizer.
7. [Galactica](https://github.com/SumanthRH/tokenization/tree/main/7-galactica/): Thinking about tokenizer design by diving into the Galactica paper.
