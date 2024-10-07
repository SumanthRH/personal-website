---
title: The Current State of Platforms for Open Source Models 
subtitle: Some observations for how open source LLMs/SLMs are being used and predictions for the future 
date: 2024-09-22T05:37:15.089Z
summary: Some observations for how open source LLMs/SLMs are being used and predictions for the future
draft: false
featured: false
commentable: true
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
It's been almost 2 years since Meta released Llama-1, with incredible progress made in capabilities, applications and recipes for open source LLMs. I want to particularly lay down trends and some predictions for the future:

- Llama as a platform (or _pretraining is unsexy again_): In Aug 2023, there were a number of players pre-training models, and it looked like that might as well remain the state of open-source: A number of research labs and Big Tech players releasing/ pre-training their own models either for it's own sake of building better models (Mistral) or tailored towards a particular application (CharacterAI). The story has changed significantly now, with consolidation around Llama as new "platform". For one datapoint - CharacterAI was recently [reverse acqui-hired](https://www.washingtonpost.com/technology/2024/08/02/google-character-ai-noam-shazeer/) by Google, and going-forward they've stated that their strategy is to actually shift to post-training models like Llama. Similarly, the capabilities of Llama, given the huge compute footprint and the talent pool of Meta, has surpassed other providers like Mistral. Two observations can be made:
    - _There will be more consolidation around Llama and more companies will switch to post-training or distilling Llama_
    - _Pretraining as a service will become less and less attractive._ There's definitely going to be lesser companies doing so as Meta itself invests heavily in pretraining + post-training Llama. This raises some interesting business questions - the stickyness of SAAS for pretraining is gonna decrease. 
- It is still early for companies finding value from LLMs: This is partly from my personal experience being involved with some customers at Anyscale, and also partly from just talking with folks in the tech scene. Most companies are still not quite sure about how best to use LLMs. The hype around AGI doesn't help either, and you'll probably see LLMs being used for tasks that they're really not a good fit for ( _For a man with a hammer, everything starts looking like a nail_). In terms of priorities, realiability is I think probably the number one, and structured output generation is the second. 
- Agents are still the holy grail: While we've seen a lot of progress in this direction (Devin et al.), integrating an AI agent natively into your workflow is still the most impactful area.
- Pytorch-native tools and libraries will become more and more popular for pre-training, fine-tuning and inference: If you've been to Pytorch Conference 2024, you'll see very clearly how Pytorch is investing heavily in building torch-native GenAI libraries - torchtitan, torchtune and torchchat. It's also very clear how far torch.compile (which, if you remember, has always been one of the, if not the, biggest focus areas for Pytorch 2.0) has come. The good thing about this investment is that there will be co-design of Pytorch feaetures for LLM use-cases, and in the same way co-design for good abstractions and performant implementations in torchtune, torchchat etc based on said features. Simplifying your tech stack towards native Pytorch will become more attractive as this happens.
- Value capture is only recently starting to move up the chain: NVIDIA (monopoly hardware supplier) is still eating everyone's lunch, but we're seeing value capture move up the chain to cloud providers/ hyperscalers. Pure SAAS plays in this space (inference/fine-tuning/pretraining) will need to wait their turn. The exception to these simple buckets is data annotation providers (Scale AI, Surge AI, etc) who basically provide human labour as a service. 
