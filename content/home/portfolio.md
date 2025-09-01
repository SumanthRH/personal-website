---
# An instance of the Experience widget.
widget: experience

# This file represents a page section.
headless: true

# Order that this section appears on the page.
weight: 20

title: Experience
subtitle:

# Date format for experience
date_format: Jan 2006

# Experiences.
experience:
  - title: Software Engineer
    company: Anyscale
    company_url: https://anyscale.com
    company_logo: anyscale
    location: Redwood City, California
    date_start: '2024-04-30'
    date_end: 
    description: |2-
      Working on the LLM team at Anyscale wearing many hats - working on new fine-tuning features, performance improvements, product CLI/SDK, etc
      * Added support for different fine-tuning tasks (such as instruction tuning and causal LM) training as well as function-calling fine-tuning.
      * Improved model support to allow bringing any HuggingFace model with any chat template to fine-tune on Anyscale.
      * Led building the [LLM Models SDK](https://docs.anyscale.com/llms/finetuning/guides/models_sdk_demo) for easily going from fine-tuning to serving on the platform.
  - title: Data Science Intern
    company: C3.AI
    company_url: https://c3.ai
    company_logo: c3
    location: Redwood City, California
    date_start: '2023-06-19'
    date_end: '2023-09-08'
    description: |2-
      Worked on the Generative AI team at C3!
      * Set up a finetuning codebase from scratch for C3's Generative Search application
      * Features: Support for difference causal and sequence-2-sequence models, ability to mix different training datasets (for a text-to-text or a causal language modelling task), visualize metrics on multiple evaluation datasets, etc
      * Trained 10B+ parameter models on 1M+ samples using DeepSpeed and 🤗 Accelerate.
  - title: Graduate Student Researcher
    company: UC San Diego
    company_logo: ucsd
    location: San Diego, California
    date_start: '2023-05-19'
    date_end: '2024-04-30'
    description: |2-
      Worked with Canwen Xu and Prof. Julian McAuley on evaluating intermediate task transfer for in-context learning.

design:
  columns: '1'
---
