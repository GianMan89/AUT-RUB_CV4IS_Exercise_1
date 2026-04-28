# Advanced data

This folder contains generated data for the advanced imaging design notebook.

## `spectral/`

Synthetic RGB, NIR-like, and UV-like images of a material/defect scene.

Purpose:
- compare RGB channels,
- simulate spectral choices,
- compute contrast and contrast-to-noise ratio.

Files:
- `material_rgb.png`
- `material_nir_like.png`
- `material_uv_like.png`
- `material_defect_mask.png`

## `line_scan/`

Synthetic continuous web / surface texture with defects and under-sampled line-scan variants.

Purpose:
- reason about line rate,
- visualize sampling artifacts in the motion direction.

Files:
- `continuous_web_texture.png`
- `continuous_web_defect_mask.png`
- `line_scan_medium_rate.png`
- `line_scan_under_sampled.png`

## `homography/`

Synthetic planar inspection target and a perspective-warped observation.

Purpose:
- estimate a homography,
- rectify a planar scene,
- measure an object after rectification.

Files:
- `planar_target_rectified.png`
- `planar_target_warped.png`
- `homography_metadata.json`

## `anomaly_baseline/`

Small generated nominal and anomalous surface images.

Purpose:
- build a simple non-ML pixelwise robust z-score baseline,
- prepare the later anomaly-detection project.

Files:
- `train_nominal/`
- `test/`
- `masks/`
- `manifest.csv`
