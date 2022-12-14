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
  - title: Graduate Teaching Assistant
    company: UC San Diego
    company_logo: ucsd
    location: San Diego, CA
    date_start: '2022-09-21'
    date_end: ''
    description: |2-
        Teaching Assistant for the Principles of Database Systems course. Responsibilities include:
        
        * Addressing student queries.
        * Handling logistics for the course.

  - title: Graduate Teaching Assistant
    company: UC San Diego
    company_logo: ucsd
    location: San Diego, CA
    date_start: '2022-08-01'
    date_end: '2022-09-05'
    description: |2-
       Sole Teaching Assistant for the CSE 21: Mathematics for Algorithms and Systems course.

        * Conducted weekly discussion sessions for 50+ students.
        * Prepared question papers for midterm and final examinations. 
        * Held office hours to address student doubts.

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
