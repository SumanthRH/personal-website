---
title: Evaluating Reasoning Models
subtitle: Evaluating LLMs is already harad. Evaluating reasoning models is harder. 
date: 2023-09-12T05:37:15.089Z
summary: Why your inference settings can screw up evaluation results. 
draft: true
featured: false
commentable: true
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---

Evaluating LLMs has already been a hard task. Evaluation benchmarks provide mile markers for model capability and quality, standardized evaluation is crucial to measure any notion of progress. For a primer on LLM evaluation in general, Clémentine Fourrier's blog is great: https://huggingface.co/blog/clefourrier/llm-evaluation 

I want to talk a bit more about evaluating LLMs on "reasoning". I've been working on the [`Skythought`](https://github.com/NovaSky-AI/SkyThought) repo recently in making a good evaluation framework for reasoning tasks. An interesting problem we ran into recently was a reproducibily crisis: We kept getting significantly results from other evaluation libraries like [Qwen-Math](https://github.com/QwenLM/Qwen2.5-Math) even while running in the same environment, at zero temperature, and the exact same inference engine settings. While some variability is expected - after all differences in batch size, etc have always introduced some variance in results (you might have seen disclaimers from the [OpenLLM Leaderboard](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about)). However, some of sensitivities in LLM evaluation are put on steriods with reasoning. 


1. Structured output extraction is hard: For many math and coding problems, it is desirable to have automated tests that are based on structured output extraction - either the final answer or the solution. This has been the preferred approach as opposed to LLM as a judge. But there are a TON of edge cases to handle, and often different papers report metrics with their favourite evaluation framework with different output post-processing rules. 
2. Long context generation at half precision can accumulate numerical errors: Numerical error accumulation from long context generation can lead to the output logprobs changing ever so slightly after 1k-2k tokens. This leads to a different predicted token and the final response for the model can change completely based on this. (the model can get stuck in a reasoning loop, the solution becomes incorrect or the response format changes). At full precision, things look good. 


These two categories of issue explain most of the differences you'll see while reproducing results in reasoning. (2) is most important, because this leads to some funky observations:

1. Evaluation results change by 5-10% across vLLM versions
2. Evaluation results at bs=1 and bs>1 can vary significantly at the same settings


<!-- While the community has been talking about reasoning - and primarily in the context of solving problems in mathematics, science and coding - for more than a year now -  -->