# CV4IS Exercise 1 — From Pixels to Visual Evidence

This repository contains the first practical exercise for **Computer Vision for Industrial Systems**.

The exercise is designed for a 90-minute supervised session and runs in Jupyter / Binder.

## Learning goal

Before training a machine-learning model, students should be able to check whether the relevant visual signal is:

1. visible,
2. sufficiently sampled,
3. stable under perturbations,
4. segmentable / measurable with simple non-ML tools.

## Exercise topics

The notebook contains three parts:

1. **Object-space resolution, sampling, and aliasing**
2. **Blur, noise, saturation, bit depth, and gamma**
3. **Thresholding, morphology, and simple measurement**

No machine learning is used in this exercise.

## Binder

Use the repository root as the Binder launch point. The notebook is:

`notebooks/01_from_pixels_to_visual_evidence.ipynb`

A teacher solution is available in:

`notebooks/solutions/01_from_pixels_to_visual_evidence_solution.ipynb`

## Data

This repository contains self-contained synthetic and industrial-style images generated for teaching.
They are intentionally small and lightweight for Binder.

External industrial datasets for later work are listed in:

`data/external_links/industrial_datasets.md`

## Suggested in-class timing

| Time | Activity |
|---:|---|
| 0–8 min | Intro and setup |
| 8–12 min | Binder launch / imports |
| 12–30 min | Part A: resolution and aliasing |
| 30–50 min | Part B: perturbations and histograms |
| 50–72 min | Part C: thresholding and measurement |
| 72–82 min | Short engineering interpretation |
| 82–90 min | Discussion / wrap-up |

## License note

The generated teaching images and notebook material in this repository are intended for course use.
External datasets linked in `data/external_links/industrial_datasets.md` have their own licenses and must be cited separately.
## Advanced extension notebook

A second notebook has been added for students who finish early or for a more advanced follow-up session:

`notebooks/02_advanced_imaging_design_challenge.ipynb`

Teacher solution:

`notebooks/solutions/02_advanced_imaging_design_challenge_solution.ipynb`

The advanced notebook contains:

1. quantitative imaging design calculations,
2. spectral / channel contrast and CNR,
3. line-scan and temporal sampling,
4. homography-based rectification and measurement,
5. a simple non-ML anomaly baseline.

This gives stronger students more challenging tasks while still staying before the machine-learning part of the course.
