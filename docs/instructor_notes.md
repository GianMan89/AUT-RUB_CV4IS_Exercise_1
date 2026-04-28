# Instructor notes — Exercise 1

## Goal

The exercise reinforces Lecture 2: pixels are engineered, not given.

## Recommended flow

1. First 8 minutes: explain the exercise goals and Binder workflow.
2. 8–12 minutes: check that everyone can import packages and load images.
3. 12–30 minutes: supervise Part A.
4. 30–50 minutes: supervise Part B.
5. 50–72 minutes: supervise Part C.
6. 72–82 minutes: students write final engineering interpretation.
7. 82–90 minutes: group discussion.

## Expected difficulties

- Students may confuse bit depth with dynamic range.
- Students may think gamma correction is harmless for measurement.
- Students may treat thresholding as "the solution" rather than as a diagnostic.
- Some students may struggle with the difference between physical feature size and pixels per feature.

## Suggested discussion questions

- When is 3 pixels across a defect enough, and when is it not?
- Which perturbation destroys information rather than merely making the image less pleasant?
- Why can a simple threshold work on one image and fail on another?
- What should be improved in the imaging setup before training a machine-learning model?

## Grading / completion idea

This exercise can be checked for completion rather than graded numerically:
- TODO cells completed
- figures generated
- final interpretation filled in
- reasonable engineering recommendation
# Advanced extension

The advanced notebook `02_advanced_imaging_design_challenge.ipynb` can be used in three ways:

1. as optional material for fast students,
2. as a second supervised exercise,
3. as preparation for the semester project.

Recommended use in a 90-minute exercise:
- Assign Part D and Part E to everyone who finishes Exercise 1.
- Use Part G as a guided demonstration if homography has not yet been covered in depth.
- Use Part H only as a teaser for the later anomaly-detection project.

The advanced notebook is intentionally more challenging and includes open-ended interpretation questions.
