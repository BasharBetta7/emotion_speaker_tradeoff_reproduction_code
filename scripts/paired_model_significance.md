# Paired Model Significance Testing

Use `scripts/paired_model_significance.py` to compare model variants using per-sample evaluation CSVs.

## Why paired tests

The evaluation files use the same source/reference/sample pairs across models. Because of that, model A and model B should be compared row-by-row after aligning on `sample_id` and, for alpha sweeps, `alpha`. This removes example difficulty as a confound: the test asks whether the same examples improve under one model.

## Tests used

- Continuous metrics such as `speaker_cosine`, `mcd_gen_src`, and `emotion_ref_cosine` use a two-sided Wilcoxon signed-rank test on paired differences.
- Binary `emotion_acc` uses an exact McNemar/binomial sign test over discordant pairs.
- Mean-difference confidence intervals use a non-parametric paired bootstrap over aligned rows.
- Reported p-values are adjusted with Benjamini-Hochberg false-discovery-rate correction across all pair/metric tests.

The recommended interpretation is based on both effect direction and uncertainty: call an improvement significant only when the bootstrap confidence interval excludes zero in the favorable direction. Use the p-value and FDR-adjusted p-value as supporting evidence.

To also run the same tests within each emotion label, add `--group-cols target_emotion`. This writes the usual aggregate CSV/Markdown plus additional files named `*_by_target_emotion.csv` and `*_by_target_emotion.md`.

## Example

```bash
python scripts/paired_model_significance.py \
  --model native=experiments/MODEL_native_intensity_ranker/ESD/SSST/full_set_alpha_variants \
  --model inferred=experiments/MODEL_inferred_intensity/ESD/SSST/full_set_alpha_variants \
  --model speaker_cls25=experiments/MODEL_native_intensity_speaker_cls/ESD/SSST/full_set_alpha_variants \
  --reference speaker_cls25 \
  --group-cols target_emotion \
  --output-dir research_mindhub/exp_wiki/significance_tests \
  --output-prefix ssst_speaker_cls25_vs_baselines
```

For alpha `1.0` only:

```bash
python scripts/paired_model_significance.py \
  --model native=experiments/MODEL_native_intensity_ranker/ESD/SSST/full_set_alpha_variants \
  --model speaker_cls25=experiments/MODEL_native_intensity_speaker_cls/ESD/SSST/full_set_alpha_variants \
  --alpha 1.0 \
  --reference speaker_cls25 \
  --output-dir research_mindhub/exp_wiki/significance_tests \
  --output-prefix ssst_speaker_cls25_vs_native_alpha1
```

## References

- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*.
