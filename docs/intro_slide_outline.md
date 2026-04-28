# Intro slides for Exercise 1

## Slide 1 — Exercise 1: From Pixels to Visual Evidence

- Today: no machine learning yet
- We manipulate images and measure what changes
- Goal: understand when image evidence is usable
- Output: completed notebook + short engineering interpretation

## Slide 2 — Tools and data

- Binder + Jupyter Notebook
- Python, NumPy, scikit-image, OpenCV, matplotlib
- Synthetic images for controlled demonstrations
- Industrial-style generated surface images for defect visibility and measurement

## Slide 3 — Part A: Sampling and feature visibility

- compute mm/px
- estimate pixels per feature
- downsample with / without anti-aliasing
- inspect aliasing and detail loss

## Slide 4 — Part B: Blur, noise, saturation, gamma

- simulate imaging perturbations
- plot histograms
- compute edge / sharpness indicators
- explain what changed in the visual evidence

## Slide 5 — Part C: Thresholding and measurement

- threshold the image
- clean mask with morphology
- find connected components
- measure area and compare to mask

## Slide 6 — Final output

- completed TODO cells
- visible figures
- short engineering interpretation
- recommendation before using ML


## Optional Slide 7 — Advanced optional tasks

- imaging-design feasibility table
- spectral contrast and CNR
- line-scan sampling
- homography rectification
- simple non-ML anomaly baseline

Use these if you finish the main notebook early.
