---
# An instance of the Experience widget.
# Documentation: https://wowchemy.com/docs/page-builder/
widget: experience

# This file represents a page section.
headless: true

# Order that this section appears on the page.
weight: 20

title: Experience
subtitle:

# Date format for experience
#   Refer to https://wowchemy.com/docs/customization/#date-format
date_format: Jan 2006

# Experiences.
#   Add/remove as many `experience` items below as you like.
#   Required fields are `title`, `company`, and `date_start`.
#   Leave `date_end` empty if it's your current employer.
#   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
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
  - title: Machine Learning Intern
    company: Hakimo Inc
    company_url: https://hakimo.ai
    company_logo: hakimo
    location: Menlo Park, California (Remote)
    date_start: '2023-04-01'
    date_end: '2023-06-16'
    description: |2-
      This was a part time internship I did with Hakimo in Spring'23. This was my first time working on video-based models, so that was fun! 
        * Worked on video-based object detection models for Hakimo's Remote Guarding Solution
        * Trained 3D ResNets on Hakimo's video surviellance data and experimented with single and muli-pathway SlowFast networks. 

  - title: Graduate Teaching Assistant
    company: UC San Diego
    company_logo: ucsd
    location: San Diego, CA
    date_start: '2022-08-01'
    date_end: '2023-03-31'
    description: |2-
        Served as a Teaching Assistant for CSE 232: Principles of Database Systems and CSE 21: Mathematics for Algorithms and Systems. Was a lot of fun, resposibilities included:
        
        * Conducting weekly discussion sessions for 50+ students.
        * Preparing question papers for midterm and final examinations.


  - title: Undergraduate Student Researcher
    company: Indian Institute of Technology Madras
    company_logo: iitm
    location: Chennai, India
    date_start: '2020-10-01'
    date_end: '2021-07-01'
    description: |2-
      Bachelor's [Thesis](https://drive.google.com/file/d/1dAyPzvIj7AUP-VrUPmmzvKc49P7VnXxM/view).
         
        * Demonstrated fast reconstruction of a 12 frame video from a single image of a lensless camera, 
          reducing inference time from 2 hours to 30 milliseconds.
        * Proposed an efficient reconstruction framework - a physics-aware neural net  
          trained in an adversarial fashion, used feature-based loss for photorealism.
        * Employed a trainable inversion layer to reverse the forward process of the camera, 
          along with a UNet for perceptual enhancement.

  - title: Deep Learning Intern
    company: HyperVerge Inc
    company_url: https://hyperverge.co
    company_logo: hyperverge
    location: Bengaluru, India
    date_start: '2019-05-10'
    date_end: '2019-07-31'
    description: |2-

        * Implemented a face detection algorithm for KYC services.
        * Trained a Multi-task Cascaded Convolutional Neural Network
          using > 200,000 images.
        * Reduced false positives 10 times and false negatives by 2.5 times.
        * Employed hard positive mining, data augmentation to reduce recall by 5%.

design:
  columns: '1'
---
