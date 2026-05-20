# Test-set evaluation (area loss, aggregate overlap, GFID)

- Real crack dir: `/home/jixi/dataset/Test_conditionalUnet/crk_mask_cleaned`
- Real aggregate dir: `/home/jixi/dataset/Test_conditionalUnet/agg_crk_unetpred/dilated`
- Target resolution: 256 x 256
- both-empty pairs excluded from means / GFID: **False**

## Headline metrics

| model | n_pairs | mean real area | mean gen area | mean area loss | mean real overlap | mean gen overlap | GFID(area_fraction) | GFID(overlap_ratio) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_clean_refer | 200 | 0.00969849 | 0.0136719 | 0.008497 | 0.626 | 0.6238 | 2.001e-05 | 0.001895 |
| baseline1 | 200 | 0.00969849 | 0.0135994 | 0.009675 | 0.626 | 0.3995 | 1.968e-05 | 0.05514 |
| baseline2 | 200 | 0.00969849 | 0.00929909 | 0.007704 | 0.626 | 0.591 | 7.175e-06 | 0.002881 |

## Per-feature means and stds (real | gen)

| model | feature | real_n | gen_n | real_mean | gen_mean | real_std | gen_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_clean_refer | area_fraction | 200 | 200 | 0.009698 | 0.01367 | 0.009017 | 0.006962 |
| full_clean_refer | aggregate_overlap_ratio | 168 | 200 | 0.626 | 0.6238 | 0.32 | 0.2765 |
| baseline1 | area_fraction | 200 | 200 | 0.009698 | 0.0136 | 0.009017 | 0.006904 |
| baseline1 | aggregate_overlap_ratio | 168 | 195 | 0.626 | 0.3995 | 0.32 | 0.2581 |
| baseline2 | area_fraction | 200 | 200 | 0.009698 | 0.009299 | 0.009017 | 0.006368 |
| baseline2 | aggregate_overlap_ratio | 168 | 184 | 0.626 | 0.591 | 0.32 | 0.2793 |

## Per-combo breakdown

### full_clean_refer

| combo | n | mean area loss | mean real area | mean gen area | mean real overlap | mean gen overlap |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 30 | 0.007320 | 0.009109 | 0.01085 | 0.7495 | 0.6697 |
| 1 | 26 | 0.009285 | 0.01461 | 0.01962 | 0.8583 | 0.8789 |
| 2 | 24 | 0.011306 | 0.006296 | 0.01094 | 0.6836 | 0.6808 |
| 3 | 35 | 0.010770 | 0.01299 | 0.01802 | 0.6103 | 0.6244 |
| 4 | 49 | 0.006472 | 0.007744 | 0.01164 | 0.4223 | 0.4452 |
| 5 | 36 | 0.007581 | 0.008375 | 0.01208 | 0.568 | 0.6057 |

### baseline1

| combo | n | mean area loss | mean real area | mean gen area | mean real overlap | mean gen overlap |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 30 | 0.010641 | 0.009109 | 0.01566 | 0.7495 | 0.5255 |
| 1 | 26 | 0.007582 | 0.01461 | 0.01336 | 0.8583 | 0.4637 |
| 2 | 24 | 0.011414 | 0.006296 | 0.01112 | 0.6836 | 0.4725 |
| 3 | 35 | 0.009549 | 0.01299 | 0.01443 | 0.6103 | 0.4021 |
| 4 | 49 | 0.010253 | 0.007744 | 0.01284 | 0.4223 | 0.2751 |
| 5 | 36 | 0.008556 | 0.008375 | 0.01394 | 0.568 | 0.3512 |

### baseline2

| combo | n | mean area loss | mean real area | mean gen area | mean real overlap | mean gen overlap |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 30 | 0.006357 | 0.009109 | 0.007848 | 0.7495 | 0.7561 |
| 1 | 26 | 0.006920 | 0.01461 | 0.01715 | 0.8583 | 0.772 |
| 2 | 24 | 0.007112 | 0.006296 | 0.005215 | 0.6836 | 0.6275 |
| 3 | 35 | 0.009569 | 0.01299 | 0.01077 | 0.6103 | 0.5578 |
| 4 | 49 | 0.007907 | 0.007744 | 0.007743 | 0.4223 | 0.4759 |
| 5 | 36 | 0.007699 | 0.008375 | 0.008254 | 0.568 | 0.484 |


## Notes

- Real and generated masks are paired by image stem and reduced to `--target_size` (OR-pool) so the effective pixel size matches between the two sides before any measurement.
- **Mean area fraction** = mean over images of `area_fraction = mask.sum() / mask.size`, reported separately for real and gen so the model's overall cracking *quantity* can be compared to the real distribution (a complementary view to the absolute-error number below).
- **Mean area loss** = mean over images of `|gen_area_fraction - real_area_fraction|`. Per-image absolute error, in the same units as `area_fraction`.
- **Mean aggregate overlap ratio** = mean over images of `|crack ∩ aggregate| / |crack|`, reported separately for real and gen so the spatial co-occurrence of cracks with aggregates can be compared to the real distribution. Images whose crack mask is empty contribute NaN and are excluded from the overlap mean.
- **GFID(area_fraction)** and **GFID(overlap_ratio)** are 1-D Frechet distances between the real and gen Gaussians fitted to each per-image scalar: `(mu_r - mu_g)^2 + (sigma_r - sigma_g)^2`. They reward matching both the *level* (means agree) and the *spread* (stds agree) of the underlying per-image distribution. Lower is better; the value is in the squared units of the underlying feature.
- When `--exclude_both_empty` is set, image pairs with no crack in either mask are dropped from all means and from the Gaussian fits (otherwise they contribute area_error=0 and an undefined overlap, which can inflate the scores).