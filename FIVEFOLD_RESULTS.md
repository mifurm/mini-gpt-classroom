# Fivefold Archive corpus experiment

**Completed:** every V1-V7 corpus is at least five times its previous unique training-byte count. The five distinct preliminary recipes were also rerun on the larger V1 corpus; R4 is the exact V1 recipe.

Each matched-budget pair has identical architecture, initialization/parent weights, seed, optimizer, batch size, completed update budget, and common validation targets. The intervention changes training text and its distribution. The control retains the original rehearsal mix; the large arm gives 20% of windows to that mix and 80% to new books (byte-weighted). Validation can select different checkpoint updates within the same completed budget; selected updates are reported below.

V3-V7 each start from the **same original parent in both arms**, isolating a stage's data change. They are not a newly chained fivefold-trained lineage. Previously delivered checkpoints and decks are intact.

## Main findings and verification

Expanded-data arms improved added-book next-byte accuracy in **7/7** versions at matched completed update budgets; differences range from +1.73 pp to +7.38 pp.

Original-test accuracy regressions: V1, V3, V4, V5, V6, V7. Legacy five-book accuracy regressions: none. More varied data is not a guarantee of improvement on every existing domain.

**Verified:** 24 unit tests passed; 25 new runs and 12 previous checkpoints checked; paired initialization/configuration/budgets match; all corpus/checkpoint hashes match; 95 protected original files are unchanged.

Each model was scored on 33,708 fixed next-byte targets, including 7,168 targets from the 14 additional books.

Qualitative inspection of fixed seeds 123 and 130 for V1-large, V7-control, V7-large, and V7-long found English-like fragments but persistent broken grammar, invented words, and mixed fictional characters. Even the longer model is a teaching demonstration, not a coherent writing assistant. All samples below remain unedited.

## Unique training corpus sizes

| Version | Previous bytes | Expanded bytes | Factor | Added books |
|---|---:|---:|---:|---:|
| V1 | 132,440 | 685,236 | 5.17x | 2 |
| V2 | 132,440 | 685,236 | 5.17x | 2 |
| V3 | 250,152 | 1,377,265 | 5.51x | 4 |
| V4 | 418,197 | 2,189,179 | 5.23x | 6 |
| V5 | 625,761 | 3,252,191 | 5.20x | 8 |
| V6 | 893,030 | 4,514,489 | 5.06x | 11 |
| V7 | 1,240,716 | 6,295,276 | 5.07x | 14 |

Counts exclude validation/test bytes and boundary gaps; no repetition or duplication counts as new data.

## Matched-budget quality: larger data versus rerun controls

| Version | Updates | Original test control -> large | Legacy five-book control -> large | Additional-book control -> large | Added-book gain |
|---|---:|---:|---:|---:|---:|
| V1 | 1800 | 54.12% -> 51.10% | 48.25% -> 49.05% | 47.20% -> 48.93% | +1.73 pp |
| V2 | 3600 | 55.22% -> 58.09% | 49.43% -> 55.38% | 47.50% -> 54.88% | +7.38 pp |
| V3 | 1800 | 59.39% -> 58.85% | 53.58% -> 56.48% | 51.03% -> 56.18% | +5.15 pp |
| V4 | 1800 | 60.47% -> 59.30% | 56.46% -> 57.65% | 53.06% -> 56.79% | +3.74 pp |
| V5 | 1800 | 60.75% -> 60.12% | 58.07% -> 58.78% | 54.09% -> 57.99% | +3.91 pp |
| V6 | 1800 | 60.79% -> 60.56% | 58.53% -> 59.66% | 54.24% -> 58.72% | +4.48 pp |
| V7 | 2600 | 61.17% -> 60.69% | 59.68% -> 60.31% | 55.13% -> 59.24% | +4.10 pp |

## Full loss and quality metrics

| Version | Arm | Dataset | Loss (nats/byte) | Bits/byte | Byte perplexity | Next-byte accuracy |
|---|---|---|---:|---:|---:|---:|
| V1 | previous | original | 1.6280 | 2.3488 | 5.0939 | 54.83% |
| V1 | previous | legacy_new_macro | 1.9174 | 2.7662 | 6.8033 | 48.77% |
| V1 | previous | additional_macro | 2.0677 | 2.9830 | 7.9063 | 46.78% |
| V1 | control | original | 1.6476 | 2.3770 | 5.1944 | 54.12% |
| V1 | control | legacy_new_macro | 1.9414 | 2.8008 | 6.9684 | 48.25% |
| V1 | control | additional_macro | 2.0608 | 2.9732 | 7.8526 | 47.20% |
| V1 | large | original | 1.7132 | 2.4717 | 5.5468 | 51.10% |
| V1 | large | legacy_new_macro | 1.7824 | 2.5715 | 5.9444 | 49.05% |
| V1 | large | additional_macro | 1.8220 | 2.6286 | 6.1841 | 48.93% |
| V2 | previous | original | 1.5599 | 2.2505 | 4.7584 | 56.96% |
| V2 | previous | legacy_new_macro | 1.8677 | 2.6945 | 6.4731 | 51.11% |
| V2 | previous | additional_macro | 2.0196 | 2.9137 | 7.5356 | 49.20% |
| V2 | control | original | 1.5966 | 2.3034 | 4.9362 | 55.22% |
| V2 | control | legacy_new_macro | 1.8579 | 2.6804 | 6.4103 | 49.43% |
| V2 | control | additional_macro | 1.9660 | 2.8363 | 7.1418 | 47.50% |
| V2 | large | original | 1.4613 | 2.1082 | 4.3115 | 58.09% |
| V2 | large | legacy_new_macro | 1.5562 | 2.2452 | 4.7410 | 55.38% |
| V2 | large | additional_macro | 1.6535 | 2.3854 | 5.2250 | 54.88% |
| V3 | previous | original | 1.4116 | 2.0365 | 4.1025 | 59.61% |
| V3 | previous | legacy_new_macro | 1.7157 | 2.4752 | 5.5605 | 53.82% |
| V3 | previous | additional_macro | 1.8721 | 2.7009 | 6.5022 | 51.23% |
| V3 | control | original | 1.4154 | 2.0419 | 4.1180 | 59.39% |
| V3 | control | legacy_new_macro | 1.7173 | 2.4775 | 5.5692 | 53.58% |
| V3 | control | additional_macro | 1.8623 | 2.6868 | 6.4388 | 51.03% |
| V3 | large | original | 1.4108 | 2.0354 | 4.0992 | 58.85% |
| V3 | large | legacy_new_macro | 1.5078 | 2.1753 | 4.5167 | 56.48% |
| V3 | large | additional_macro | 1.5235 | 2.1980 | 4.5884 | 56.18% |
| V4 | previous | original | 1.3787 | 1.9891 | 3.9698 | 60.47% |
| V4 | previous | legacy_new_macro | 1.5812 | 2.2812 | 4.8607 | 56.46% |
| V4 | previous | additional_macro | 1.8017 | 2.5993 | 6.0601 | 53.06% |
| V4 | control | original | 1.3787 | 1.9891 | 3.9698 | 60.47% |
| V4 | control | legacy_new_macro | 1.5812 | 2.2812 | 4.8607 | 56.46% |
| V4 | control | additional_macro | 1.8017 | 2.5993 | 6.0601 | 53.06% |
| V4 | large | original | 1.3883 | 2.0029 | 4.0080 | 59.30% |
| V4 | large | legacy_new_macro | 1.4544 | 2.0983 | 4.2820 | 57.65% |
| V4 | large | additional_macro | 1.4907 | 2.1506 | 4.4400 | 56.79% |
| V5 | previous | original | 1.3499 | 1.9474 | 3.8569 | 60.75% |
| V5 | previous | legacy_new_macro | 1.4760 | 2.1294 | 4.3753 | 58.07% |
| V5 | previous | additional_macro | 1.7236 | 2.4867 | 5.6048 | 54.09% |
| V5 | control | original | 1.3499 | 1.9474 | 3.8569 | 60.75% |
| V5 | control | legacy_new_macro | 1.4760 | 2.1294 | 4.3753 | 58.07% |
| V5 | control | additional_macro | 1.7236 | 2.4867 | 5.6048 | 54.09% |
| V5 | large | original | 1.3548 | 1.9545 | 3.8758 | 60.12% |
| V5 | large | legacy_new_macro | 1.4063 | 2.0289 | 4.0808 | 58.78% |
| V5 | large | additional_macro | 1.4324 | 2.0665 | 4.1886 | 57.99% |
| V6 | previous | original | 1.3444 | 1.9396 | 3.8361 | 60.79% |
| V6 | previous | legacy_new_macro | 1.4281 | 2.0603 | 4.1707 | 58.53% |
| V6 | previous | additional_macro | 1.6991 | 2.4513 | 5.4690 | 54.24% |
| V6 | control | original | 1.3444 | 1.9396 | 3.8361 | 60.79% |
| V6 | control | legacy_new_macro | 1.4281 | 2.0603 | 4.1707 | 58.53% |
| V6 | control | additional_macro | 1.6991 | 2.4513 | 5.4690 | 54.24% |
| V6 | large | original | 1.3448 | 1.9401 | 3.8373 | 60.56% |
| V6 | large | legacy_new_macro | 1.3640 | 1.9678 | 3.9118 | 59.66% |
| V6 | large | additional_macro | 1.3973 | 2.0159 | 4.0444 | 58.72% |
| V7 | previous | original | 1.3492 | 1.9465 | 3.8543 | 61.23% |
| V7 | previous | legacy_new_macro | 1.3991 | 2.0185 | 4.0516 | 59.21% |
| V7 | previous | additional_macro | 1.6689 | 2.4077 | 5.3063 | 54.63% |
| V7 | control | original | 1.3445 | 1.9397 | 3.8364 | 61.17% |
| V7 | control | legacy_new_macro | 1.3847 | 1.9978 | 3.9938 | 59.68% |
| V7 | control | additional_macro | 1.6672 | 2.4053 | 5.2974 | 55.13% |
| V7 | large | original | 1.3326 | 1.9225 | 3.7909 | 60.69% |
| V7 | large | legacy_new_macro | 1.3391 | 1.9319 | 3.8156 | 60.31% |
| V7 | large | additional_macro | 1.3703 | 1.9770 | 3.9367 | 59.24% |
| V7_LONG | large | original | 1.2708 | 1.8333 | 3.5635 | 62.36% |
| V7_LONG | large | legacy_new_macro | 1.2539 | 1.8090 | 3.5041 | 62.04% |
| V7_LONG | large | additional_macro | 1.2559 | 1.8118 | 3.5109 | 62.37% |

## Predeclared longer-training check

V7-large continued its live model/optimizer trajectory from 2,600 to 15,600 updates (6x the original budget). The multiplier is the ceiling of corpus growth, fixed before training to approximately restore presented bytes per available byte. No test score chose the extension.

Additional-book accuracy: 59.24% -> **62.37%**. Legacy five-book accuracy: 60.31% -> **62.04%**.

This extended result includes additional optimization as well as the expanded corpus; it is not a matched-compute estimate of data improvement alone. With a fixed update budget, increasing available bytes reduces exposure per corpus byte.

## Preliminary six-experiment replication

| Recipe | Corpus growth | Updates | Added-book control accuracy | Large accuracy | Difference |
|---|---:|---:|---:|---:|---:|
| r0_baseline | 5.17x | 200 | 28.45% | 29.99% | +1.55 pp |
| r1_more_training | 5.17x | 600 | 38.50% | 37.68% | -0.82 pp |
| r2_longer_context | 5.17x | 600 | 36.94% | 37.28% | +0.33 pp |
| r3_more_capacity | 5.17x | 600 | 42.42% | 41.85% | -0.57 pp |
| r5_scheduled_lr | 5.17x | 1800 | 46.44% | 50.11% | +3.67 pp |
| r4_train_longer | 5.17x | 1800 | 47.20% | 48.93% | +1.73 pp |

## Checkpoint selection, compute, and training exposure

| Version/recipe | Arm | Updates | Selected update | Seconds | Presented MB | Bytes presented / available byte |
|---|---|---:|---:|---:|---:|---:|
| r0_baseline | control | 200 | 200 | 0.7 | 0.20 | 1.55 |
| r1_more_training | control | 600 | 600 | 2.2 | 0.61 | 4.64 |
| r2_longer_context | control | 600 | 600 | 3.3 | 1.23 | 9.28 |
| r3_more_capacity | control | 600 | 600 | 12.6 | 1.23 | 9.28 |
| r5_scheduled_lr | control | 1,800 | 1,800 | 37.5 | 3.69 | 27.83 |
| v1 | control | 1,800 | 1,600 | 38.2 | 3.69 | 27.83 |
| v2 | control | 3,600 | 1,600 | 304.8 | 14.75 | 111.34 |
| v3 | control | 1,800 | 1,000 | 150.1 | 7.37 | 29.47 |
| v4 | control | 1,800 | 1,800 | 150.0 | 7.37 | 17.63 |
| v5 | control | 1,800 | 1,800 | 196.0 | 7.37 | 11.78 |
| v6 | control | 1,800 | 1,800 | 194.4 | 7.37 | 8.26 |
| v7 | control | 2,600 | 2,000 | 279.4 | 10.65 | 8.58 |
| r0_baseline | large | 200 | 200 | 0.7 | 0.20 | 0.30 |
| r1_more_training | large | 600 | 600 | 2.2 | 0.61 | 0.90 |
| r2_longer_context | large | 600 | 600 | 3.3 | 1.23 | 1.79 |
| r3_more_capacity | large | 600 | 600 | 12.6 | 1.23 | 1.79 |
| r5_scheduled_lr | large | 1,800 | 1,800 | 37.5 | 3.69 | 5.38 |
| v1 | large | 1,800 | 1,800 | 38.2 | 3.69 | 5.38 |
| v2 | large | 3,600 | 3,600 | 304.9 | 14.75 | 21.52 |
| v3 | large | 1,800 | 1,800 | 150.2 | 7.37 | 5.35 |
| v4 | large | 1,800 | 1,800 | 149.9 | 7.37 | 3.37 |
| v5 | large | 1,800 | 1,800 | 196.3 | 7.37 | 2.27 |
| v6 | large | 1,800 | 1,800 | 194.3 | 7.37 | 1.63 |
| v7 | large | 2,600 | 2,200 | 279.3 | 10.65 | 1.69 |
| v7_long | large | 15,600 | 15,400 | 1295.9 | 63.90 | 10.15 |

Timing includes periodic validation/checkpoint overhead and excludes final test scoring/sampling. The two arms ran concurrently, so wall-clock timings are not isolated hardware benchmarks. Window sampling with replacement means presented/available bytes is exposure, not exact epochs.

## Paired uncertainty over added books

| Version | Added-book accuracy difference | 95% paired-book bootstrap interval |
|---|---:|---:|
| r0_baseline | +1.55 pp | +0.91 pp to +2.22 pp |
| r1_more_training | -0.82 pp | -1.70 pp to -0.01 pp |
| r2_longer_context | +0.33 pp | -0.49 pp to +1.28 pp |
| r3_more_capacity | -0.57 pp | -1.62 pp to +0.52 pp |
| r5_scheduled_lr | +3.67 pp | +2.51 pp to +4.99 pp |
| v1 | +1.73 pp | +0.53 pp to +2.97 pp |
| v2 | +7.38 pp | +5.61 pp to +9.46 pp |
| v3 | +5.15 pp | +3.74 pp to +6.71 pp |
| v4 | +3.74 pp | +2.47 pp to +5.26 pp |
| v5 | +3.91 pp | +2.59 pp to +5.18 pp |
| v6 | +4.48 pp | +3.24 pp to +5.64 pp |
| v7 | +4.10 pp | +3.01 pp to +5.22 pp |

These intervals describe variation across this book collection, not training-seed uncertainty or a representative population of all writing. Books share authors; this descriptive book bootstrap does not account for author clustering.

## Generated-text quality proxies

All models use the same nine prompts, seeds 123-131, temperature 0.8, top-k 40, and 240 generated bytes. Lexicon matching uses one fixed dictionary from all training documents, never test text. Neither spelling match nor low repetition proves grammaticality or coherence.

| Version | Arm | Fixed-lexicon word hit rate | Repeated word-trigram fraction | Mean UTF-8 replacements |
|---|---|---:|---:|---:|
| r0_baseline | control | 65.71% | 0.000 | 0.00 |
| r1_more_training | control | 66.35% | 0.002 | 0.00 |
| r2_longer_context | control | 62.16% | 0.000 | 0.00 |
| r3_more_capacity | control | 71.99% | 0.000 | 0.00 |
| r5_scheduled_lr | control | 86.47% | 0.000 | 0.00 |
| v1 | control | 86.00% | 0.000 | 0.00 |
| v2 | control | 90.49% | 0.002 | 0.00 |
| v3 | control | 96.13% | 0.000 | 0.00 |
| v4 | control | 98.43% | 0.002 | 0.00 |
| v5 | control | 96.61% | 0.000 | 0.00 |
| v6 | control | 96.05% | 0.000 | 0.00 |
| v7 | control | 98.37% | 0.000 | 0.00 |
| r0_baseline | large | 55.00% | 0.000 | 0.00 |
| r1_more_training | large | 62.30% | 0.000 | 0.00 |
| r2_longer_context | large | 58.62% | 0.000 | 0.00 |
| r3_more_capacity | large | 64.14% | 0.000 | 0.00 |
| r5_scheduled_lr | large | 79.60% | 0.000 | 0.00 |
| v1 | large | 79.67% | 0.000 | 0.00 |
| v2 | large | 93.48% | 0.005 | 0.00 |
| v3 | large | 91.94% | 0.000 | 0.22 |
| v4 | large | 93.23% | 0.000 | 0.00 |
| v5 | large | 94.84% | 0.000 | 0.11 |
| v6 | large | 95.97% | 0.000 | 0.00 |
| v7 | large | 95.36% | 0.000 | 0.00 |
| v7_long | large | 96.91% | 0.000 | 0.00 |
| r0_baseline | previous | 65.71% | 0.000 | 0.00 |
| r1_more_training | previous | 66.35% | 0.002 | 0.00 |
| r2_longer_context | previous | 62.16% | 0.000 | 0.00 |
| r3_more_capacity | previous | 71.99% | 0.000 | 0.00 |
| r5_scheduled_lr | previous | 86.47% | 0.000 | 0.00 |
| v1 | previous | 88.12% | 0.000 | 0.00 |
| v2 | previous | 94.13% | 0.000 | 0.00 |
| v3 | previous | 96.58% | 0.004 | 0.00 |
| v4 | previous | 98.43% | 0.002 | 0.00 |
| v5 | previous | 96.61% | 0.000 | 0.00 |
| v6 | previous | 96.05% | 0.000 | 0.00 |
| v7 | previous | 98.17% | 0.000 | 0.00 |

## Data, rights, and contamination checks

Added scans prepared: 14. Their pre-1926 narrative texts/text editions are assessed as public domain in the US under the [Cornell copyright-term reference](https://guides.library.cornell.edu/copyright/publicdomain). Some Archive metadata lacks a binary rights flag; its actual statement and the separate date-based rights basis are preserved, not fabricated. Some scans are later reprints of those public-domain texts, not physical pre-1926 editions.

Acquisition rejected an undated Fauntleroy candidate in favor of the dated 1886 scan, and a damaged Treasure Island candidate lacking its final narrative paragraph in favor of a complete copy. Neither rejected candidate counts toward training bytes.

| Work | Edition | Clean bytes | Train bytes | Source |
|---|---|---:|---:|---|
| Little Lord Fauntleroy | 1886 | 323,774 | 259,019 | [Archive](https://archive.org/details/littlelordfauntl00burn_1) |
| A little princess; being the whole story of Sara Crewe, now told for the first time | 1905 | 367,222 | 293,777 | [Archive](https://archive.org/details/littleprincessbe00burn) |
| The railway children | 1906 | 350,157 | 280,125 | [Archive](https://archive.org/details/railwaychildren00nesb_1) |
| Treasure Island | 1900 | 367,740 | 294,192 | [Archive](https://archive.org/details/treasureisl00stev) |
| Black Beauty | 1905 | 310,049 | 248,039 | [Archive](https://archive.org/details/blackbeauty00sewe_10) |
| Heidi | [1910] | 494,788 | 395,830 | [Archive](https://archive.org/details/heidispyr00spyruoft) |
| Anne of Green Gables | 1908 | 573,679 | 458,943 | [Archive](https://archive.org/details/anneofgreengable00mont_11) |
| Anne of Avonlea | 1909 | 495,632 | 396,505 | [Archive](https://archive.org/details/anneofavonlea00mont_5) |
| The adventures of Tom Sawyer | [pref. 1876] | 394,905 | 315,924 | [Archive](https://archive.org/details/adventuresoftoms00twaiuoft) |
| The Adventures of Huckleberry Finn : (Tom Sawyer's comrade) | 1912 | 571,517 | 457,213 | [Archive](https://archive.org/details/adventuresofhuck00twaiuoft) |
| The jungle book | 1899 | 277,365 | 221,892 | [Archive](https://archive.org/details/junglebook00kipluoft) |
| The second jungle book | 1915, c1895 | 349,544 | 279,635 | [Archive](https://archive.org/details/secondjunglebook00kipluoft) |
| Little women, or, Meg, Jo, Beth, and Amy | 1915 | 1,025,620 | 820,496 | [Archive](https://archive.org/details/littlewomenormeg00alcouoft) |
| Kidnapped : being memoirs of the adventures of David Balfour in the year 1751 ... / written by himself and now set forth | 1907, c1905 | 416,213 | 332,970 | [Archive](https://archive.org/details/kidnappedbeingme00stevuoft) |

Metadata publication dates can refer to the original text edition, not the scanned reprint.
- princess: 1917 reprint of the 1905 narrative; title page inspected.
- railway: Metadata says 1906; title/copyright page says 1905.
- heidi: English text edition first published 1910, reprinted 1944. Only that public-domain narrative is extracted; front editorial matter excluded.
- jungle: 1899 uniform text edition; reprint list ends 1915.

Audit: **12,238 heldout 128-byte spans**, checked against every one of **6,292,736 document-local training offsets**; **zero exact matches**. Rolling-hash candidates are verified byte-for-byte. This remains a strided, exact-byte audit, not exhaustive semantic deduplication. Every book uses contiguous 80/10/10 partitions, 128-byte gaps, checksums, and document-local training windows.

OCR mistakes, dated language, and historical stereotypes remain. Check local rights and classroom suitability before redistributing. The common added-book test includes books not trained by early versions; all versions nevertheless use the same target positions.

## Protocol and limits

Same targets, all versions/arms: original 512, other books 128 each. Selection loss = .25 original + .25 legacy five-book macro + .50 additional-book macro. Check every 200 updates; seed 2026.

All paired and long selections finish before performance testing. Original all 16300 targets and five legacy books 2048 each reuse exact prior targets. New books: 512 distinct targets each, seed 2032, equal-book macro.

One seed, changed genre/domain distribution, within-book holds, imperfect OCR; V3-V7 use the same previous parent in each arm, not a newly chained lineage. Byte accuracy is not coherence; fixed-lexicon sample scores are spelling proxies.

Tests and checkpoints are never chosen by test results. Legacy holdouts were reported earlier and are reused, not fresh. New-book performance was first opened only after every paired and extended selection finished. One random seed cannot establish a robust causal improvement; larger corpora also change genre and token distributions.

## Reproduce

```bash
python3 fivefold_data.py
python3 -m unittest -v
python3 fivefold_train.py --arm control --device cpu
python3 fivefold_train.py --arm large --device cpu
python3 fivefold_train.py --evaluate --device cpu
python3 report_fivefold.py
python3 generate.py --checkpoint artifacts/fivefold/large/v7_long.pt --device cpu
```

These commands overwrite only the new fivefold experiment outputs. Detailed per-book scores, all curves, paired intervals, hashes, and every sample are in `artifacts/fivefold/comparison.json`; flat metrics are in `artifacts/fivefold/metrics.csv`.

## All unedited fixed-seed samples

### control / r0_baseline

Seed 123, prefix 'Alice said, "'

```text
Alice said, " ! 'The s say the bre an tin we aly s d he " thanitond f anal t thang w le arid oug s lound ghm I ty werer d  lyouint t selphd thit hind y ou, he wn s "hinisThan ce thed the ig th thinous ce s The lduid s the thed tous tbed tove the m poug 
```

Seed 124, prefix 'The Queen '

```text
The Queen t talis aikin s tht they t topu shin ashid y hothe Hin f tofis s shicto s he oucang as t wald, we oon o t he thiwhe, t ooul, r wakelis t on hanled und, koling  fth than quguced ad anghed ay t it thawouooullld tid t thius, w wan we tout t st
```

Seed 125, prefix 'The little '

```text
The little y hthe wher ay and the ame ke Que ol ooas ce oul thin her hen w thex: "


" ove " ca t ceedned t whengut,an here wi, t'tn n'g
'to he t d she tou s" thsat the he heveng s ! " s " " he-athey theshing l tond wouoke th t the  ind cedou antt t t
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, " s he d as ashof t tugh t herat : 
,444 werin t bling ay an ar aner ced shed athe Aly t lyid thowit oke weas Hu s ant fofouathe s nd d atls wee beb th sthers bait s coorohan aw ceve t maved l as ere loun thad ed id me meery t hacit thed sea
```

Seed 127, prefix 'Wendy '

```text
Wendy ricke ta int r she igoued t the has m s ve e an' f ayoaly y bupet the ty aaneg it w ci: " " tho whas t t " thire" m, s she o be I wapot hers terou, t t d the lou arof w t hither bin nlimalyNok  beere ou in'e t bee heit coly aloor t out s ae
```

Seed 128, prefix 'The Mole '

```text
The Mole ourer llin f shel Aing mon heke s ten the segheay a s t he soound d beney th t. "



 thicicich bu s hed t, we gup thes of don ale t. "



" nouthar Angouts allds t : Doe chet see at t mote onghasall, s fid to w.war I " Nom oos gouis t t p 
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox d oh Owit thin Wouid lit  be there thalild wo t ther wis t sos td p ounr an veed oproche temrong t t s ! 


" Af d rane tout het soul t t weue her che fowais ang we d bidof juthhe t yo inghy won n, Sherinid


44 I I t veerit d chef men't  t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " ne se an t At torinlipd ou heny me wont and k.The t thed Save t chou t : 
" lid t ithed ald an tu thid a theor se  t d weithod sind ing ou an tnte tid rancon t t ke but.  Dof py Aloraib, shisn ches t arem t ower f, wo f hremose o wals and 
```

Seed 131, prefix 'The children '

```text
The children the ay," m ano w e f vitofon s at prou sh, t thitt  ske ? The tyonge thens t t ther nd oathiwitt ir caa woly nes. " whin In ke t s he na calin  sin co ad rowit s wint. 

" ad So s bing wouling and Al ? Ovand s in ae t a d muthe ain t thang 
```

### control / r1_more_training

Seed 123, prefix 'Alice said, "'

```text
Alice said, " she I gring ate br she hand a they sto he " I was oned. " and it and wy mas to ought lhe hegrmme tham and do knoting itse puple it his, you forr ones ear it hand and edree sim to thisue ble she saidd was eand herspe shebed to wer hemppst l
```

Seed 124, prefix 'The Queen '

```text
The Queen tro wer : une sat thelly u to ther," and thy kothe wing too if said at aste boucte sar toe lddo ea wnas to to could, it oulver the wishe of has and if the her the lot very uump anke it anay und it awnd ould ousid she iupte thand eet the lit
```

Seed 125, prefix 'The little '

```text
The little for a cull way," wen the cor ourd on oof  cherte thing, " hen in it : "

" It thelly kne tone it wecnt.


" It if iu to nin's 'to hers domy woong "


"It the he hryou the be so rouge sae you ont he lllong anto bing an he finge on't and sthe
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, " she ad could of theult tooneat : I,444 woothing lite and wand an reemp she said and it leve thoumxto she as the fand foly woon son the blst the wbe has wass and you oon hand wer she smave. The sers and and !" I id mut seeap the ceats the a
```

Seed 127, prefix 'Wendy '

```text
Wendy rith itt in ar she bethed the was anm she se an't muy and you and you and and in a Qui: " " though sing the gart the st but eellt ? "


" " It bout the dche so ther ousste the by it noom woish abe to ous wee to the bitte lroand and as the e
```

Seed 128, prefix 'The Mole '

```text
The Mole onger los fforsed And to and knit the doughes the a ste fe soo ne toulde uft the sood thoung it but ther the fpe she hellf dor an afd the'h dnoown a ongeststly troun said and ase wat nas the queasaid, shem cre wown't do hom oos gook an at i
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox do. They hentupl sid and a the sim ther an wown and awnd awnd ! "

4 Kne an thers, worne ther in the said to-boor so he to. Bult so.


" Iffue ner could was to to thelf oom jutthe ther ingey wor net morsside.

4 I wher olld a wandf moootn t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " Thelle han at jors inppent anny mponed waside." Hut the you onn and it : 

" So the greanld an turn noth faror se it downithoersing it bout ootnte tie ranco whey kno th. "


" And ong anke won oh so to so the ea frt too hrem se oppatst yok
```

Seed 131, prefix 'The children '

```text
The children the gy," I' so ween and of nos and rought, " thiss aske ? " I wallly the sQueeenter. Alice thimte and and she corsaid in and and so fing anl il sit cor the withe mint. I whas th she be woonger and an ?"


"Its in and ound muscme in them wol
```

### control / r2_longer_context

Seed 123, prefix 'Alice said, "'

```text
Alice said, " son the sany the bruid had the aly she said il a ompeaf alally but on y mas id ough sthe he dmit tham a ppriter : Hon't you don ith ith you manthen sailly. Thand to hedr a sid the sifore crea mes ittille ea veny said itbectto in have pof l
```

Seed 124, prefix 'The Queen '

```text
The Queen troully piking at thouly u to the up as it yourthe nin for? is said at asthe oucan qusaioutly, weas tho the said a ist oo lonr the wisth of hatllly nof you you lthe : and dobed a bruthe say und itt.


"Eillght it she iupten was we so the mu
```

Seed 125, prefix 'The little '

```text
The little for a out she and she ta coke Queen. " I vereraled in hermppoowng exkittugh ove : an the yon dot wacnly lan thiseng, then sary't she sid she tor said said he as hy she say." said Ale-sasty a ond he b ton hat ombing at he you and hisand sthe
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, " she ad as as of the geged, ean : ", pranen, 4 inl at ay sthe Queer a I she said ally the westly.


4 Alitas the faing of pathe sin ? " the was llo the : And and sain it had awachely, mand. T as ere a dnothed ep ittle bre a t hagin't sthat 
```

Seed 127, prefix 'Wendy '

```text
Wendy rith say deen tshe it."


the I on calll queain't bu therey busht you and and poom Qui: she and wiss the saidd the nos she oe llittly, mierse aboutht loucke, an ith wow the a stouin noom with rabe to on gane toree histhlly mawathe dose a a 
```

Seed 128, prefix 'The Mole '

```text
The Mole cerer lof aflently in to a hake hat ging heds this of elle soo ne douldednty you's the toully, all sthe : she an she the fad noom at. "


" And thar ond Alitly the sally, and asee att a preatte ssaid, shaid to waway as illlly. She is the yo
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox din a a the as souid lit a the sabe the lot wat ther wit a sthe ddloow. "


Shed oprecte thir in the said toly wersaing torthe thupsed uriffuelyy here fobais a t we dlit of jutthe the binghy wouth, she soid


44 I whes oly the matf moooth t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " The lis ! " the Shallpente heny miced that demored at aby ave tof ther : thellllt ith Ud said the : nou a shoond, it saily, "dand I it boutte thee prithatuld thad ske the preas and aa brith snoorest torem the ea fre thor reetse on atstore 
```

Seed 131, prefix 'The children '

```text
The children the gy," I as skee frain'f nos aid rought, ist ifth she ? the "


There as if the wand oat thimthir ca a sthe lorsaing it ne and she gon cal il sit imp astooneds, she you't don rithing with sip noout ?"


"I shing aftuled mutfmeednthe mawal
```

### control / r3_more_capacity

Seed 123, prefix 'Alice said, "'

```text
Alice said, " said Alice. Hall drumpt turthen, you do bectiles oment. " I " wehat no loose doong stoo nothim I that's puck it : Ho they phing it his, you mant wass ! " Alice to It got theim to this ore wen The imumpted Alice said. " becaling the not to 
```

Seed 124, prefix 'The Queen '

```text
The Queen trough tought says ! " aid to jest," said ther to his if to ifirst imtomsten ouch, quicke ald to fight ! " said Alice too loor the wishe only quace if the has tot loook a goot asher hiss alll

to bawousE to old the was aye thand ever up lad
```

Seed 125, prefix 'The little '

```text
The little for a began ! "

" Alice couldn't slooking are linting, " Alice to Alice oliove : conte a deedd awacn'ut, and if the twon thro's she say. you'd it helse for the hras tremever but he ! "

" You Alice. Mongs byoce the fight is a I his wenly i
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "baster. " I can't thugh tooked by I, pran it it blie fay as argain the ! Ahice his to the imper said to ifast if it then cult — never the was begin a :


" You'ss could and wace lot marked ? Marst alll to ! " said Alice to that thes tas a
```

Seed 127, prefix 'Wendy '

```text
Wendy rathing but a tshe is in ? " Alice call very we' be you felll and the pristery it wors: for and whas to the grre" then she oer cit.

"things terow, then chen sallled she the cloout onle alll drabe to one weent it's bee ! " said Alicese Alic
```

Seed 128, prefix 'The Mole '

```text
The Mole of tright forst," Alice on happed ten to help this of the she hads were the ! " she said, it's curpred dot, wore ! "

The con'I llow. "An't dent sar the Alice ! she were'lll the pisten tote onle sall, shaid to hew you litttle sage in the pi
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox dexp take this souid little brod be to alll way ther was a said down't reat to coourced one rounds to : thim that say."

She ? thu slow then elimst.

" Alice as ! Alice I didn't the sallly ? ""

Cast Distider to I was intten that fallly for
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " Alice said Alice walking way ne first that demore that.

" Then frost them the the flidd shat's uca ndry face as a treatent off ind in in it one enow your on that she tro preap. Alice buten seach somented too ea fre too hre toe olld soonlk
```

Seed 131, prefix 'The children '

```text
The children the gyet that she down tof the for the she tat it head ? " The you case asQueen. " Ind idow whet ? what wold ther and in and and roment are willett impadsing ! " Alice salll. She heis said mereand Alice ! " ther the to erem to fain them

Al
```

### control / r5_scheduled_lr

Seed 123, prefix 'Alice said, "'

```text
Alice said, " I don't sany you have a turn it, you know."

That misself all obout not make to the floud, because asken," she said, insectly this high one for a must hrough and to her the inventides of crwarm sile?" said Alice.

" You've a birthd.

"I sh
```

Seed 124, prefix 'The Queen '

```text
The Queen cross at fun one other air to them, and it crying and if they is she to stuon the mean off loud, and so that the began openerring when you know."

And the Unicorn lot seemelbody her himself under always, looking at this that, so ever the mu
```

Seed 125, prefix 'The little '

```text
The little for a little panting to me ! " Alice said in a little her hand of them turning !"

444 Here had a ment.

But the with the sarv'y shoutid ? " the Unicorn, the kedd here smove her. Ahow as you long. " It's a looked her fall on and his stoop o
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't know ! "

" Well, the found her, if you could not have the boat's a joust that, and she was wanted to and for patting never the when the other way, just come have were lyorm?" said Alice : " and they wanted to you have a shorse,
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a detorn, this in the Queen : " the ear head yearfully and the proud.

"Not drifmished of the kinds quired.

" A botherel of pot herself.

" The very sent all of the poor bit neard ! " said Alice, when she thought the wood-by sett re
```

Seed 128, prefix 'The Mole '

```text
The Mole offtrorted them, and she said, if the firesely as she called of the broud.

The little to did, and strong," Alice said, after and shaked one for her only them ! " Alice excited, peating out one off the roow to cown badler, and the blacked i
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox do. They hadn't fun the end beg, because I would," said the King, poe fread to come of both to say the stating for ague to them turn waus went on her, for a faul turn," Alice : he there's as and never the head off abellott, as a jam with Me
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " when I had an joke in down a ne minute offide."

Alice began to cry very manage to him. Alice the whille face as a tried, as Alice with him up nied to carrie that she thought shake as affer she has not more one of the find.

" I shall seem
```

Seed 131, prefix 'The children '

```text
The children through the so with was sileng at provisice as it

as the treest of him as he tremp.

"I showieth your a of face."

" I'ven make a bound reamily," shouted, without Alice what the waite with treen that she chose, and the mimselfot ket that l
```

### control / v1

Seed 123, prefix 'Alice said, "'

```text
Alice said, " So I don't you to way that the back to be " Sheep may from people it would you like floar midly

that they's just HUSE,"

" I look hill you manishess ! " said the Rose say ! " she said, and said, as it saying himsely : " that wish] the Wal
```

Seed 124, prefix 'The Queen '

```text
The Queen prost all unhardly ! " said the Unicorn imply, that is for? "

" I didn't have can't set as the garden ! " said Alice, openes to speak herfly, " then she said nothing very leplace, and as she could.

" Evly on it at a looking of elt on the 
```

Seed 125, prefix 'The little '

```text
The little for a little pan, you saying over on to the wouldn't like another ' Sair of the way other best themcn ut, and then I tell sare's she was she could, that couse exheed : " you suddenly-gake an roll. " It not good in to the corne I harant stoo
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't can't be getting at : I, plenty it into his sandwichied at Alice was the brilling bother know as Humpty Dumpty was he next cribsily like of the sod just come hard with growing to day : " and it wased it much ertes hage stick ! "
```

Seed 127, prefix 'Wendy '

```text
Wendy I can talke an she been do it way and she comes he watering, and the corse are it would you'd tell of better dright, and like lovered to best about the dentle other Rulling, and been that with a teme one weet of the wood lower of them. At e
```

Seed 128, prefix 'The Mole '

```text
The Mole of tright with just look any kring ently cally as a ste fell offt there they cares time to the clape, whethere explained, and I liked the hidner ! " And Alice bested had that see waten to large with there.

" But I'm sure I shok bad an to s
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox day our the insers, the came the better ablewing, and while been down treed to comply way, it in to corksetily. " However, the thought uris on the couldn't be ! " And that's dream that in the right Square asking ! " the King remark long for
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " I'm afraid Alice's lips one only so one : it's the panted at another sure

" So thing made any dear nothing or see to beyin off ind in the ash nast ime as it that she thought shanking butters to spoking one of the two or rememed spars andw
```

Seed 131, prefix 'The children '

```text
The children sew and the so way faving on side of the had thing asked on her lone the spoke to fand that with it calless jumped away as see the ball arge in the counts, with gling you've got it it as Alice horsed is ! "

" But with ernes fect no sharp l
```

### control / v2

Seed 123, prefix 'Alice said, "'

```text
Alice said, " She was spoke," Alice was said in a tone to them of the King said in a little King silen it immement.

"But the picked seemed to the which : " then so her something her the ide's brokown, when her must of Alice was and because it a middle 
```

Seed 124, prefix 'The Queen '

```text
The Queen suill as suddenly that a smy put him as it your for this ! You're give as he couldn't hold like a mone ! "

" I was a moon vore a which sort as as und, you know the gave touch are ! "

Alice went on, as she with a was one of a more out the 
```

Seed 125, prefix 'The little '

```text
The little for a little pance of them kitten on to be a sween in her hand-bun the trow oves the top with other name, and the but off the dard of the horse's and sation the hill off ! " she said, "but the Alice. My I ways bark an moment. It's anstant o
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I'm going of the good-now the does and side blieg an an armoment to she said, as it last the better wors with an ! Face the side didn't like about him, so she said " had awn beling any the Messenger, that enought you're to the stards of
```

Seed 127, prefix 'Wendy '

```text
Wendy I can talke an she been deal the way was queere's be talking her the care anything Queen she round of the would got be she other house, herself on, the screame there's quite clossie to make it aberway the seemed with the looking at of the e
```

Seed 128, prefix 'The Mole '

```text
The Mole off right : but the Tiger-lily.

" I can't know," said the Queen. " Alice was said to this, if you see ! "

" Yes, went of don't think," she said, as she water tried on eith the piate. " I should a little."

" I was a little one back the pi
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox did you know, and it's be been siger all be with any we've sistide, and the voice procted, thoughtfully the Queen said, to the top suddenly on the country ask him up, she strow her handing she hands more in

on, and the top King said, with 
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " The little was

"I little very happened again."

Alice began an ground : and for she god said, " My not began puzzle sittion a mind in ? " she said, as as in that something on beesonat for wher her finger the ears, which remosed sparsson t
```

Seed 131, prefix 'The children '

```text
The children the great in of the was of things pround, and it whisped struch a beginning of the same as things?" cried the Queen, so her kind round a chork off the Queen, she was she was one the be wood sidenly with the kitten aften the sheen kitten wil
```

### control / v3

Seed 123, prefix 'Alice said, "'

```text
Alice said, " So you should have think it with you down " She had saying " Miss isn't makes to go fly. They much as as he were interesting me out in, you might must have out your hear ! " he said, over went on in a sudden tone. " You've to be one the se
```

Seed 124, prefix 'The Queen '

```text
The Queen cross and she spoke, " and then shut of it your for you do? "

Tweedledum cried and brooken to her head to herself. " So You might here, of couldn't be kind through the queer-large undersation. " The Knight is it wasn't only a feet of them?
```

Seed 125, prefix 'The little '

```text
The little for a little panter of the kitten of the song with her, and while the Queen this time, when he said not large in the tear sort's sharp down at all, that it's all. My Miss May butter, and you know."

" Come out bird ! " Alice said in a sugge
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see it's go to crab : I won't think you're you. I was a little directly with you to be to see as if I wish I can't have to leave the good do?"

" Yes, if it you like to be mady."

"Mind, and they're time one come that you can ea
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a ching shopite of the wood, and seemed to be there, but the King said, " and it was are what the sizes took of the other head, Did you might to see it much, you know." And whether the road ! " the Queen cried to the Duchess. " Not a
```

Seed 128, prefix 'The Mole '

```text
The Mole carer little she was queer her.

" I can't know," said the Tiger-lily, "you're sometimes anxiously — you know, please of don't think it's drown arm at all makes in his time a sort that tell you, and shaking my were the road."

" She's a v
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox delight the Suppose to be the glass was looking about with the glass with her head round there somethings to the dark.

" That's look a little gently," the Knight remarked.

" It's a right," said the Duchess.

44It be very on large more fin
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They must you jave in delight you, would be a creature.

Please it said "

" Some it much particular of bushrook, it didn't perhaps it ? " said tell Alice : " and so the Dormouse to a bite wonder so marked and it her look turned aways agai
```

Seed 131, prefix 'The children '

```text
The children that was made or enor insect side the word of them asking the way, that as if she wanted away, this called by the best broner the while she walks too go such a shut

which a little back quite pantion in the kitten. " You never heard not to 
```

### control / v4

Seed 123, prefix 'Alice said, "'

```text
Alice said, " So you should like that the window ? "

" Well, it's a part of the great Hatter."

" No, it's my come and do not it I know how you had a good near how he were eyes they to size off. Took crossed all in saving his pocket-him. I do have not 
```

Seed 124, prefix 'The Queen '

```text
The Queen cried and sixty-the bees usual

shut as if your found if you silence."

4* 1 never saw the little girl ! " said Alice took very suddenly of his head. " You know the gave you want be all say to the Tiger-lily ? "

Alice wouldn't see what lit
```

Seed 125, prefix 'The little '

```text
The little for a country.

" The same a tire of this way with it hand: which ' you could tell you call you meann ugly and yet, and not going her down upon its mouth the hedgehog."

" Why, it wear something be a little good really May invion't as well 
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see it's going, eat you, please, if you're you. I was reached his arm," the Messenger said, whispered the followed his middle.

"The Ball I don't believe in that away fly," Alice replied and looked at the Mock Turtle. " Is that 
```

Seed 127, prefix 'Wendy '

```text
Wendy I can talk to the big."

" Well, what is is the house ? " said Alice. She had pine this, she found of her head them, and shoes came to the same of the Lion.

" However, you know because ? " she struck the sea. " And we could not cats quite 
```

Seed 128, prefix 'The Mole '

```text
The Mole of tried with the night of her.

" I can't know," said the Tiger-lily, "you're sometimes and fearfully down with plum-cake. They were ten in the proming upside the brook, and went on a good open said, " I'll go twice things of the big, and 
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox day ! " the Scarecrow. " You may be a very body want with you, you know."

"I shook come to to say the Tin Woodman

and then ? " said the Lion. " Only for a game with brooch ! "

" Really is more so many very lonely than the mean," said the
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They had not got to be all her hands at all."

44 That's the Red King said, like a shridgle of the girl, for you, it's my own a paw in the air needles came to any skime. Dorothy Alice began snore so sore of course, which rememember this wa
```

Seed 131, prefix 'The children '

```text
The children through the soldiers violent start round, and then she could not make a nice tone of the horse's pricket, so she was so very long when they well in the dark, and with her hands right her.

" Then I would have dinner to them ! " and the Gnat
```

### control / v5

Seed 123, prefix 'Alice said, "'

```text
Alice said, " So you should like that the window ? "

" Well, it's a melancholy of which when I can have got to your for?" said Humpty Dumpty, as it as she stood spoke, "That's all draw ! "

" It's only what sometimes to have spoke," said the King.

" I
```

Seed 124, prefix 'The Queen '

```text
The Queen cried at full six that a presents, and his hardly hand for their shartless were some shoes like a conversations way, it was srat sweet as they lay under they grot looked at bellow and fastened.

The Cap, loating a way upon than we so until 
```

Seed 125, prefix 'The little '

```text
The little forgetting way, for she must of the books from the bellowed with its minute or two, and then the wacnought and stuff then she had had done.

" You have to his eyes," said Alice : " was speak on my name ! " Alice replied. " Not I heard the G
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't sent to until I eat you, please, if you'll you like to the bit's a pirate that may of the shapes were all followed him. Alice screamble to him, so she said "That's the last," said Alice, " and it's moment here and hand at all. N
```

Seed 127, prefix 'Wendy '

```text
Wendy I can talk toges to get to be along myself eyes, " I think I have you are any pirates : you'd tell of come to really story of the way," he said, "indeed cut of them of still they in them, Niih ant this came out of this would look at seem an
```

Seed 128, prefix 'The Mole '

```text
The Mole carefully for the night of his ment. However, the song the slowly way, he forts thing that it had been ditched to her stood no one that wouldn't have no upside to like a chair eyes to her eyes, and this side of the made of her hands and the
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox dech that was some of please the boys of the Cat, and was well tell me ready.

"Not can one round much ! " Alice asked, to the time.

44 If I shall do not any my well," the Knight said in a very new more very good-bye, " and what falling it
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They had not good all down and his eyes and the Poor

words of the shrieks ? "

He was quite excution, and hopesed to begin with no it, and so needles raturned at last the boys and rather her to see to me the end of the house either story 
```

Seed 131, prefix 'The children '

```text
The children that was the other five of the door, and he stood heart on the top of her piece.

"In your face?" said Alice.

" The King, we must be a pretend for the dark, who was were talking her hand, meeping away — I shink I don't must have to my be
```

### control / v6

Seed 123, prefix 'Alice said, "'

```text
Alice said, " So you should like that the window ? "

" Well, it's a plan forth of green ! " said the Rat, my mouth and down this, which had somehow, you may comply here ! " the Queen replied, as soon creature it as she could speak to clasp it again, an
```

Seed 124, prefix 'The Queen '

```text
The Queen (who was funny at the mark town shut of it, that he could of sight it out her lister.

" Of course it out of the beast, of course," said the Caterpillar, looking at the queer darker interrupted in a sun, " and it wasn't oping of ever the da
```

Seed 125, prefix 'The little '

```text
The little first of West."

" I suppose that she's me a sulky," said Alice : '* and seems to be always like a cneedless to find then I'm

just as she said, "I must be be a hungry small so many ! "

" You how to the same came ! " he said.

Alice took a
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see it's going," and if he would hardly have a little girl been so much a thought to herself : she had no found the sent directions by the greatle and stood up his eyes other shall says, and her feet in me of the whight the same
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a catch stopped him up his head, and they had talked by him the road.

"Oh with: fyou moment it seem to them," the King said interestly. " She musch help her feelings flying and many of the most crose to the bit of the other of the e
```

Seed 128, prefix 'The Mole '

```text
The Mole carefully for the night of him.

" I can't know," said the Tiger-lily, "you've seen it wouldn't be seen before, if you don't make a bit of conversation," the Rose said with a seemed to him, " of saying this time."

44 Don't you go by the ba
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox dech that was some of pleases gently without quite eager and stiddly under the cool of both road, and sat they were guessed the turn was the end of the twins and were find the room. They had not never their

old no best, the much fame out o
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They make you join in the way your now ! "

44 And you'd been no preventice of their getting out ! " Alice asked, in sixtion (as she said in a needles rather patiently, 44 believe it's far why it's a prementice of the first the others and 
```

Seed 131, prefix 'The children '

```text
The children that was the other friends, which she was, taking his prisoner to herself at the moment and watching a great clear curious of the white presen cape out of the Cowardly.

The Mouse was silence properly sight to in a face dimbection tone, whi
```

### control / v7

Seed 123, prefix 'Alice said, "'

```text
Alice said, " I should have to be a comfort than you ! "

She had saying " My has no use ! " And she cried.

" That's one of this time it doesn't in," said the Munchkins : " I wonder if you 'll try out would be done I'm afraid it."

" It's the most of y
```

Seed 124, prefix 'The Queen '

```text
The Queen cried at last : " that's the pudding of it your finger with is back too stay."

" That's a little girl ! " the King said, looking without fluther out in great returned and looked round at the Great would have been to fun, and she saw up the
```

Seed 125, prefix 'The little '

```text
The little figure who was a wrist among through the horses and sleepy, while Alice could help standing, the picked on the best, then she 's she stood. "It's a good long away as I can do believe to part of my blittle and curious they were day and the G
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't know ! "

" Well, the voice of the relief, you know," said Alice : " when they may provokes it seemed to get out."

" Oh, if I'm a soldier something like our hands," said Alice. " You're going on repeating the time things have a
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a cottage.

He heard the horse she seemed to you remark him that that you make have to take out their dream. You complain how we herself over the child of all she was quite eagerly, as she was still explained it would have been the e
```

Seed 128, prefix 'The Mole '

```text
The Mole of tried white standing on his ment. He faced him a stepped of next.

Then the still finger-pieces were bed with shuddenly dreadfully to see him her one stream to leave in the end of the telescopsant, she was so way as if he had been for hi
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox de. They had been into the thing began all way to find small so dull and go very opened over on them the Mole of the Pawn, the turns was the end of the two branches of the little house in the room, smiled as if the best, and we had made for
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They make you join in the way you of the girl."

Alice began to first the

LOOKING-GLASS INSECTS.

" Of course it is it," said the King. " You needn't remure that it wouldn't have been invented his minute men ! "

" I did you expect," said
```

Seed 131, prefix 'The children '

```text
The children that was made here friends, which should have got her place on the counter.

The mounted and went on a ground, so and she remarked her a choke off. The Lion would see he was a bright covered it was a little kiding after and the window, and 
```

### large / r0_baseline

Seed 123, prefix 'Alice said, "'

```text
Alice said, " p sitig shay the hand whand w they shed s " isashemesaf anas t bur ond le arid oug sslound gimig thamerer d  ly tint hese phere ithend, ore meer wass shinisThan ce thedrer sig tl thinous cres mes ldund s e anghed wofor bed tovid he n pous 
```

Seed 124, prefix 'The Queen '

```text
The Queen tr amere fune sethth pry u tupurshin ashit y hothe ang f to isis shirto sthe oucan s she wald, we hinas tind thiwae at oos f r taswhes ason hanled und tke he g lth than angucon whe ulind as t he walwounErllad sit s thiusthit, s we touud ld 
```

Seed 125, prefix 'The little '

```text
The little lo tas thor ay anor hed me ke tre ol ooy  cheralenghe her hen he n hamitug athiche s tleredeld t wacngealan here wi, te n sarl'to hego d she toris" thsat the he heve t s sached raree sas y tanon he l tond an ocbing ar he for. I She anttli t
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, " sshe d asitoung putule t herat fid, pmanerinof bling ay an ar anere to shed athe t t the imanlorextoke weasr t se an fofopathe s nd d atlse te blb th stherlowan ss  oorohan awe sshe smaved l as eshal d herad ed id me breryo thag theshe sea
```

Seed 127, prefix 'Wendy '

```text
Wendy rit asta te ar stofig." Ke The thas med inee an' bed talyey beret the arisaneg it werit, s taro wisof ll at ghre" mins she o bllithe wt hers tarou, t t ache t chine oud te theshouin n amal tid  bere sote wee toute hesthe ly awa hard se ad e
```

Seed 128, prefix 'The Mole '

```text
The Mole cererills ffushel n TN ay he kriss pe dan seshas sa ste he soound ly canduth t. "



 thishingh bu — d th we gupl anedof d noulerid the'hed thes a theluts a bes le lane cheteeee atthamot long wsalr, s sit to wawariad grome hingouis t t pi
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox d oh tate thin s sid lpe  be theres the lo wo th, ng wis awe s td por he an veedeoproche terround much sthid tof asr he tont het soulw ulingoe her che foware an owe d."TI st d athet so ingis wor n, Sherinidego t so ben perit me hef menotf t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " he ale h tyot jorinl pdend heny me ad thand k. Anot thed Save t chos tthe
" lowet ithed ald an tu a whe bushor se it d weithodanind ing thias hnte tis rarul w at ke but. Lore py Akerat Mishisn ches t premoman er fe we f hremose oshatston k
```

Seed 131, prefix 'The children '

```text
The children lerig as mid os we f vitofon s at proutou, tat itth sk ch t haty michin; s " t the Ind mathiwiet iricaa wol f cerlan s bron e berr herou cke il sin co adsrowis s lint. Loutad So s bing witheer and as ks tashkshin ad to ad murf te k tnthangl
```

### large / r1_more_training

Seed 123, prefix 'Alice said, "'

```text
Alice said, " Sorraig stay the his and in with you dof the ham to saf and to the one litg to ong ssthe he dmig tham and do he sid I kse phin soom ith you wast wiss she is had canthed the im th bek oot cres mes lound ste the you she toectto with so poool
```

Seed 124, prefix 'The Queen '

```text
The Queen tro were furn sat the maid to the un and thy hou he in furof sird gittllde onotcen quranid, sunde int olding the be, it on for. " " Saras the slag und too he mrotillght andlbed a be his say une this say," Wad sid stoe unople, " wess and lit
```

Seed 125, prefix 'The little '

```text
The little Farrince on apand sun tame kittre old," she astim. " I she mon sto his tho pove wach to then dot wicng. " "


" I dunde nin's 't she sid she to ist the coord he herent saive so ror eas." I ndont he litth said crith and Minss. "



And the s
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, " she ad as as of theult to ceat fit, Sid, oin the said yollly ham re to she said the it leitttlo he or he as the felt fold at — necacke swite lible a the soand sse in ohar awac she smany. T as essalld her !" Ming med se and hag thes bace 
```

Seed 127, prefix 'Wendy '

```text
Wendy rith itt intte she is indet the havem shinged " he dyou sey bedettye a stare geitttlri: — " " wis it sthe gdre" mins she on llit. "


" "said on, " I dche slag arrys. Bevichth bie them woNok ant to ste weeors, was stoo lis woorced seead e
```

Seed 128, prefix 'The Mole '

```text
The Mole The ris said she and old and krifften the sestack a ste feryoound douce yof werstilld this it, alll— doth we gup the dof don an sta the he nofth a ondests a bes in stoe ca the histte tore the with ther it to wewas as wors ooseged said thi
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox di. The the is sound Fabe be Fasirsther lothe was ne wis toexstyd por he and a cooplce bone toen the sres told the rand torthe tou ste ullde elly he sof's a the walld."Toom juthen ther inggs wor n, She soideg boong be he tit th hef men't Me
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " he sle han was arinld doow heny miced thand kmoth bett.




" Eardrrrrs: 
" his this gid he and un' whin ther he witth wenthoer ind in bout boonted it caruld thard witt. "


" Eand ham Mishiss Me sI I rem cande butt to in the ea sparstandk
```

Seed 131, prefix 'The children '

```text
The children the a and ham sare fraits on stard to tou, " shis he knof the he micking. It the momery and witth bich and the srand with ne and somed a callig sid cof the withs litt. " Cead So she dent. I st and and so and shing aftul drett me thound the 
```

### large / r2_longer_context

Seed 123, prefix 'Alice said, "'

```text
Alice said, " she I griss Mrlis Laid haus with y she had to was ond fon a to the hew lous id ofgh foher Shim I tham and do his int hese phim hith is, you mant he scell is aid certhed the im he fainght cres mes idund she him you sher bectto wis and and l
```

Seed 124, prefix 'The Queen '

```text
The Queen tro what fure sat that at mauptesting arid. Arot he in for theis shim omsthe ouchn qus touald, wea whas the sthil, aaid of for waswe she of has and he fove you fth thaver gust a her his sab unet waswoused aaid it ; and and thand esto un lit
```

Seed 125, prefix 'The little '

```text
The little Faitincelors.

" He sat men woure old," she astlen he her haw her 'sking ow ove wach theredn his wacntug,and the wi, then sarl't she she she tof sthe sace his Mend she smeve sour hie was y aanst he litth he wor ith aith he saived ikand sthe
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, " she ad as as of betugh stherat fid, pranering inling ay and I and thed she sat ind to theim tho he oosh oas the fand fof pating cne dintly was blo th sthe soblitin coor had awackely, many. Thas ess a dif ondees id men fery tou gitste besea
```

Seed 127, prefix 'Wendy '

```text
Wendy rith itt int hes of gounde at wa — the qued " heve therey busetty and thand pite shi: — and whas the his dreat ins she oe llit. "

Did shim outht he cher lous he swat his stouie noom wor he beet iste ane the wavert oo rma a hat as thine
```

Seed 128, prefix 'The Mole '

```text
The Mole Thickilld hevemel ning ay haver hat gid fomes this of en was of ther cand the sstid if thad if alind, wast, wene she he of Lond list. "

''che hen a ond utstlabls in hise chimese watth mote onge said, shaid to waward dour has sagoois a at a
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox do. Thay has uplfuictled a thigh bent oout wo the ne wis a sting tho whe an veersopleche her oing mmas in to the dor he tont he turstwe the oe her cherstwais and weroust of juttht ther ingh ver ; she he sidegon, I wher wat he wand her 't Mo
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " he sle had whe at al ponow hen his wout her kmoth bett.

Save tof he sthe hil wothish ha ld and uch whis fator wer this lith das no it fould oonte aid dancing and a but. "


" Amess aiblishis a hes aid mout o ea fre tho hrem seios alstoulk
```

Seed 131, prefix 'The children '

```text
The children the gy ther soy wen want fon sthe pronest, ast ive ask chat hatt mit the speft ther nea at thiet. " Whares chels. I wa Lof se Andr has a cknok you shous sund ! she " whe wam and she as wis merpens ait she ouls in a faun dom the ain her his 
```

### large / r3_more_capacity

Seed 123, prefix 'Alice said, "'

```text
Alice said, " she sight nevery had found the have the was there mand. " I sone a grey meant ther stoon therm I that and do lin in theseep don it his lore forthen scearties an certhed the in ther quouth were here withe exeven the her becallking and to gr
```

Seed 124, prefix 'The Queen '

```text
The Queen the with fure sot ther a dry pupshing a it yourthe wich dow isishe in ondeer ouch, das ton litttle whas the scome, an— sheer was wishe do his and her knever the lother willl and a lit sab the fit. Cendricame it ; and as leeand ever un lit
```

Seed 125, prefix 'The little '

```text
The little fortaious whopen would amoxe on a strooy  charwere her,sher which 'mam thow over the lit Cinddishiscne. " Aneveer in then ther's shee dolly would thersece his wend she sa very dround-stappinerst went ther would and ther her fonder and the s
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "bust a nos as not thugh to cramby d, prand, in she a fay litttle hre worked sof the cortheim thow — shee sruch fant ford at — neverything told it her woldd this coomeht. Gow she the verythan everlfung ondeer it mun feee wereght the sa f
```

Seed 127, prefix 'Wendy '

```text
Wendy rittle a deveryst about det the have findered n't muy alfor beeen were sthaned it with dis the with it ther grreat inst but everie. " Inid sot bo her lonck.

" His she stimes somuch not alllike beeffious weeor weer."

"Only a morce oun a le
```

Seed 128, prefix 'The Mole '

```text
The Mole Misss Mis His mugh in to her kressten the segrver a steeped loones wiched ther. "I dig that 'me litttle the wellish shee fid nonle there ficke it a bregan starkes were welll theer at the tellp towall, shaid to wown't down here was in ther w
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox dexply. His a safures. The there ben would would, never, and sot deeen. " Anve coould weould on jummpt ! Missss bearguan on the ture was the elitttle, for a faun were find the a grownd in the roth, Sheris a gon, In seev, and we hearm a they
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " he sleapperst joricusponss a ny minnd that demore the would she chought word forthis god like the Eand ficked heer ther litttere. I much with here sid cancin that skeee. " If shee sas buch was she fit ther and inf the fin the evoing songer
```

Seed 131, prefix 'The children '

```text
The children the gon the so would visof hes for the she was it hesken sthe was were ; shee them and man thim. Thick a slinces. " An Lost Fanterome, and her you scomed, I dons, shee was a lit she chere, mereand as know shis in a ficked must feen them wou
```

### large / r5_scheduled_lr

Seed 123, prefix 'Alice said, "'

```text
Alice said, " he ! " she said or shall is with you do because," cheef said Mr. Hobbs lowere sound slown them I can't all only time ; seer pen it his lordshing with eartic of the the rooom — and book, when he loudd stori the spossioned to with the stor
```

Seed 124, prefix 'The Queen '

```text
The Queen troum any unhappy that you to them," said Sara. " But if theree arginteds.

So can sarrose loudden that the schoool, it of throus which souldenly turn, and the fthatt her gustler-boorss say under the door a disippers!" she said, as out the 
```

Seed 125, prefix 'The little '

```text
The little for a course.

" He had man expressed," she said, "becheed, whille that could he like they all the cnaughan had for, then sarr's should like to it, and there are young smean sudden ever, and his head tone whoo bitte and her and Sara. It the
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but the could n't tough the rather of man of it blied and want and therest's a pretty the were one— she asreed if the found — never they let sbet done which was compoht. Gway fly mmarr. That exple dixcimeden in the feet torbacts, and ea
```

Seed 127, prefix 'Wendy '

```text
Wendy rich it. He were somether that the child quester for theree begginne and shee pind shild she forgoasince and sreated his like lower pot herself over the child never of she Earl you and many in abellow theree to certain to roan of theree, an
```

Seed 128, prefix 'The Mole '

```text
The Mole hastried a from to bellaning upiss enture seemed the telle scount out, he for storious that if all seend to me gues and that that him the humpered a little talkes legs becater eyes to her eyes triage, and whoo cown take him.

" That you a l
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox wirply. I will sound little blast — and the that that, and so dolle from to him, would never something at her best up to them to seee to contry her, for a faut were find that he ther anything ; and her vaining to be very the chaf mom the 
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " he sleap you to that poor and you people fire."

And they ave to pretty low lowed it god like the Earl one horse, it difly. "And no in a little telling and the first feel people explain.

She how so to could nearfe whick rempser sparsions.
```

Seed 131, prefix 'The children '

```text
The children little simple so enound of his and round herself.

" It ? the begging and poling to him and with it calless followan in hand her some and had loved hous, roous she had been done she chair, mer and as is vasulsting aftually, after-noon and l
```

### large / v1

Seed 123, prefix 'Alice said, "'

```text
Alice said, " Sometimes and ther with the with you down " if a to say ! and white no lows to ought the hoping more drepudid.

"WHAS CY.


peneved it," said Mr. Has said somed called row side looking theres mer lownds, and and she fatted to beturned of h
```

Seed 124, prefix 'The Queen '

```text
The Queen troame to she sat that you to to him and the know hich fat? if she is ofsed boucing as the looked of a him. She was a toos ver taking the fact lactual, — and for than quilstanched himself und one.

And lowed it ; becustan thought out ther
```

Seed 125, prefix 'The little '

```text
The little for a could by any the apperetty, old, as she we was liked monenting: but those was to crying it wacnlet, and then I to not go'rls of dold. How is, mised the he had nothney heard him and hand showedl tone why ching able her face his whose a
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see " think it, eat you," man risitinly."

" How I puppted befuld them to the interows to she as the fact for pattiagent ding bewles.

" She was beautifully one."

He slow many. Thes ever fant one expidence ferest hach a shessed
```

Seed 127, prefix 'Wendy '

```text
Wendy rick it. He wersterighted to him. The stalled had be talked bedut her prester. " Sara : "I think you to bernis throng to be the Itteret," she about thought.

" His," she was to think to dowN IO beer some wer to te him to look a hard seem te
```

Seed 128, prefix 'The Mole '

```text
The Mole cortried quite stand of of his man privanted had been ev ery of therelfed.

Theselled thought, alwas heart, with pled stor don one that of. " Then a one upstarits light eld puted catter of some what ther mind of.

That her hows would not ha
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox who have her beartive. The things staring the to beg with ot so duppon. " They comproced teld in the sagratic pould gues of the turnew upilf-elder coments as a to minues, and they ther beath wors storning her books best, and what fact out t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " The leave you to have done a new pond to girl."

At the fish and preven: they wet for havy a his ching a pacerne, in difticultaning in the almonged in camulant frock them pretty more to her sitting his eyes.

" You well how toes and some t
```

Seed 131, prefix 'The children '

```text
The children the great han she deaving on suddenly ; he was intered come on there to him. They was a an things bick and becture child breat throught and he countion thrown to him when't down in in the Earl on the know ouls in a fact mome for the stant l
```

### large / v2

Seed 123, prefix 'Alice said, "'

```text
Alice said, " I 'm very not like that a street you ? "

" I 'm voice. " I could n't feel and help for not impulay were adored the India," said the Roson, " I considea with a low the room in the first bedwarm said in speak the speak of the visitage poor 
```

Seed 124, prefix 'The Queen '

```text
The Queen trouble after a tite provus appearing a struth.

" He is a disible into see boy any as the little first to take when it wells in a little fire ladies forget a long than and confessions. Sara was a little little face and splendid as if it wa
```

Seed 125, prefix 'The little '

```text
The little Fauntleroy's panty an earls are a slow as he saw thinking her head of the water walk at the ladies can.

But the window-night into her dress of the most court were short some burning-room, and Alice. Minchin one the big had jected and talke
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see her until her hands, princourt in the ays way and relationation which the instrong-room, and How it a foiruption on the black eyes. " I don't believe it is not quite the visit as every difficulty is much a time to the childr
```

Seed 127, prefix 'Wendy '

```text
Wendy I can talked."

So it is to the boat window a new may alred begun to carriage them things of the said —" she was not solemn like a work, something to see her, instead ; "the boy wanted with ratter to come out," she thought was a deside of
```

Seed 128, prefix 'The Mole '

```text
The Mole of trifle for her side of her red the day were a suddenly used near before the string this come before the face, she had feet only the good deep and one seemed to her and children attic. I wonder, and she was so way."

"No," said Sara, as i
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox did not the curlous top a bear simple — that Alice was a sort dull bered to him, and be eyes into a little boy as as if it had somewhere how he had an' are a good dufferent. It was a very very new more in good earself, and we had may to t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " nearly of you just like out one mines? I 'd know some about you care to look at the great day the Earl of Dorincourt difficulty in the boy and needs a rature thought Lottie.

" There is," said Mr. Havisham, " Jabbs well, I muse to be to li
```

Seed 131, prefix 'The children '

```text
The children little bread. " I don't see think provish, Sara's has grandfather in the spirit. If I don't want it called them, and if I never saw again always in the dark, and with her handso making with the next skmall has in a face mamma more of the gl
```

### large / v3

Seed 123, prefix 'Alice said, "'

```text
Alice said, " I 'm against the brush of the things ? "

" I 'm voice. " I " said the little King Phyllis. " She had not do not it I know how him his lordship. He said to That captain the instant; "oher were he looked at all. As she took to with the stor
```

Seed 124, prefix 'The Queen '

```text
The Queen said ; " Our the children, you're uncomforty to find mission."

" I can see in can dare," said Alice.

" And the began of there are is a delighted under of the ftorth presences of purprisag under the door was silent," said Mother, "you thin
```

Seed 125, prefix 'The little '

```text
The little Fauntleroy's pany of them. The remork, as she went on her hand stopped.

" It is that?" said the Lockarge.

" Well, Sara, not 's she was so looking to at the hedge," said Mother. " Havispage you were to fancyour the door, in a while world o
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see it was getting by the man of it would less. I was recovered so much a thought long to she ask the pind of path his made blue the long again some wind on on his cheeks, made already she did.

" I " said the King Sara. " What 
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a catch stopped her eyes of the piece and fellowered by his man's saily in the house, and the squeer garden inst but everything minds, and her lovely sent as if she had been and made it abelled to meet out the trool of the footman an
```

Seed 128, prefix 'The Mole '

```text
The Mole of tried quite standle and had been glanched that he telle hearned to me.

" I should think if you see," the Gren runned face ; " If I got up it ! " and Alice lost in a great deep at the tell of said, " I'll every one of the hought see at a
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox who have had been into the things — — the table brow and so dull the good and proper there something at the bast under. But that was the elder cheerty asked out of the little house in the room, Smollerty.

“ I think it’s jam with Mo
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " I don't know I could polour just some to give him to her have to preven at the beautiful passed up him." As or secomple time as if it was a book eyes in the white skind. "Dick you see are her to see to me me that you look to see in the gre
```

Seed 131, prefix 'The children '

```text
The children like your hands were violed to do provess, and it’s a great by mick.��

“ You’re made anything a grown scorning of any and round a chool countrous,” said Mother, “ it’s been something that she called to find my before not that D
```

### large / v4

Seed 123, prefix 'Alice said, "'

```text
Alice said, " I 'm a good and bring out of the year?"

"Well, it's a manage obstic when she would be and gimme that I don't you will see him on the day of the country with a low the room in the squore, when he looked at him to poor the glove across of h
```

Seed 124, prefix 'The Queen '

```text
The Queen cried and she sat thought, and then the idea hot her life. You're give as to go any as the little of a great cup and the sleep was like so little out for the grate of pretty what being as you had been out of it was an untain, of even until 
```

Seed 125, prefix 'The little '

```text
The little Fauntleroy's pany of them. The removed as were leaving, and we want been on the walk at the horses; now in a pretty, toon the door of the house, and heard his feet shore saddly round and can long. He took at once her from in a last and stra
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see a puult into a big, and were it would be anything relady the other stitting and began she asked in a whole way his mide blue the lange stonely, and then one of the little bedle March and the does in me of the three starts of
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a dead this before the way and still a she way already and the prisoner of the house, and the grandmother to be stuffered that this sorry child no me, and all she was quite explained in a truck to weet out the trotteman of the statio
```

Seed 128, prefix 'The Mole '

```text
The Mole carefully for her nightening to him. He could not her eyes soon to will you were some time in his party before, in the doctor of the mountain of his called to be deep and chatched at the tears they carriage to her first, and he would go tha
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox declothy had been into the things — —I won’t go womanswer down the goats cool convered, in the same to-day as as I could tell me up found you meant at the great fire that he thought is worth, Smolly as if the beside it was jamment fro
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " That's a number own by to me al me," he answered. " I don't say it some long," Just he did not be a new bushed weare difficulty in the boy and not time as if they she did not say to a bit of nothing to me to tell you, for you ever also, an
```

Seed 131, prefix 'The children '

```text
The children leage and hands we friendful start to him, and if he knew the by misster. Queen were papartioned in called the sazanding on the room.

“ You can spok as I’ll might be taken I have not got pany and so as their as the doctor of not the gl
```

### large / v5

Seed 123, prefix 'Alice said, "'

```text
Alice said, " I 'm going to the run of it with you down " She's myself to be buright like to legs, looking might as I put my friends seem down out the boy, and not she went on to her to sight about her weeks still when Alice was a robed to see her possi
```

Seed 124, prefix 'The Queen '

```text
The Queen cried and she sat thought, to them, and it was to be mind. You're given some out and shook like a convoident to when the same went on the blue of the fire hand of the pretty words and just tunk on away of the side when she was feet out to d
```

Seed 125, prefix 'The little '

```text
The little Fauntleroy's pany of them. The repole, as she went on her head over his collor.

" I know when them now in the day unterning with her dark. However, he could have her dear cheek," he said, " You will think of course you have a been and stoo
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see her up to him the door and things over you. I was really the other stitting and saved she asked in the old world on the blue tears of his most and second had away flying very day exclaimed.

" Who did you can that you come a
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a descressed to her expect and she seemed to yourself, and you are and pite or herself."

" I'm so much that," said Anne in a white sorroof, " I don't like to see me," said Marilla, "if you meant it everything to them."

Anne seem to
```

Seed 128, prefix 'The Mole '

```text
The Mole carefully found the Tine Woodman would have got up on the shool.

" I must have to be this in his pretty to me, Mrs. Havisham I leave the hurried, and the starber were because to catter of long with the fir to cown made him. Mother said, �
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox de. They had been into the things — — the man next did not the work of the cook of both rounds the station. Raches I could tell me up found you meant and a good deal of just the song again, the Schin is good by the people waitful of fit
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " The leave you joy all down and his worth fire."

" It was a very prevently place," she added Clara. " No used the children, perhaps in the air not at the conversation the pretty next more his John, and so many asked to go the earl and seem
```

Seed 131, prefix 'The children '

```text
The children least and has sure friendful star proves, and Diana sat down the mick of speech. If no make a misses a grow following of any and some on the ill in the dark, who was were taking in his woodmen and as if they she could not must her head and 
```

### large / v6

Seed 123, prefix 'Alice said, "'

```text
Alice said, " Sometimes all the run of it with you ? "

" I was myself to be but I was a great girl."

"She's my ame and they said itself pleasure in your," replied.

" IT is not here," said Alice. " And she said, " I say you see you've to be called to 
```

Seed 124, prefix 'The Queen '

```text
The Queen (" all the Queen stood a pretty shut of the trotten in from the child of the old had says:

But the was time the began of there are is a different, and killed the lottle wild not be right being on away out and beautiful of that we could not
```

Seed 125, prefix 'The little '

```text
The little Fauntleroy's panion. We must that she's secrettled in her head of the bought his face, and the same next, and then I tell they're heard up. However, heard his feet, and my cheeks of a few and shout of the wood, and there were very and the G
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see him up to him the door and six in the descrations. There was a piece of levest of the sheeps to see the fourth and did not with blows again.

"I wondering and we call," said the Cedric did.

" I says, Miss Marilla I didn't k
```

Seed 127, prefix 'Wendy '

```text
Wendy I can talk to the big."

" Well, — the keep " Wendy ? " said Mr. The Goat.

"Oh, drivessed to what's the side," said Jole enjoyed to Diana. " Then you can see it, you might just be took out of the sound ? "

" And it would look at see you
```

Seed 128, prefix 'The Mole '

```text
The Mole of the little mounting a short.

" I can't know," she expected the Green Rottenmeier is the peep, " that they had been to come all the book of the room," the King said : " but it watch one of the air, so it is well bad that once by some tha
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox de. They had been into the things — — the Hatter would see your do. I have come of both round my pretty crowded and the middle of the feore you decided with great find the room. And I have no several in good to be very time infrom yef t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " That's a number own by the pony mistake and kind of the fire and prevently the bed, Peter day's grandfather?" she could not hear now.

" What is every dark the first ? " said Praying, but Swanting on a mouth wearful off her toes and store 
```

Seed 131, prefix 'The children '

```text
The children that was the old enough of the door of she was indeed the trees. It was as if the fland as they will as work he reached to be kind her a chool off. The dark, who was were taking in it, and merpense to say outside and can not feeling shake l
```

### large / v7

Seed 123, prefix 'Alice said, "'

```text
Alice said, " You don't say there if had the same to be " She is a dead an orth of her Harrison. Why, not imagination put me for the sea hall of his long fast on scent in his captain. " They'll be one creature it?" as she replied. " I can very much just
```

Seed 124, prefix 'The Queen '

```text
The Queen (who was full so standing, to the uncomfortunation and for the basket of the cock, and then she didn't like a countrain of three. She has been lady as from the ftoot: " And be a great faste in the awful," said Alan.

" It's was well just as
```

Seed 125, prefix 'The little '

```text
The little Fauntleroon Harnia had a cording polited him out and lessons of the baby of the wall of the house, can glad him and put on again, he was so long to the countrar have to my cheek. I was speaking on the while to be he was a prized of and stra
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see he'll the horse fir, prancision blows and wanted in the shoulder and brilling lost to she asked if it would the send disture the long door, so I was not only away from me. I like every day. He expected the time things he sea
```

Seed 127, prefix 'Wendy '

```text
Wendy I can talk to the big."

" Well, — May keep " I way interested."

I 've a both, with myself of the glass, * You'm not but you I wanted," said Tweedledum, " You could see the boy in the world of the steam-ent of the trouble."

And the stre
```

Seed 128, prefix 'The Mole '

```text
The Mole carefully for this blood and keep the day well about the way, had done.

" But she did not like a pale," the Queen said, for the Queen, " if you had got up to be dear about the other family only said, " I'd tell you about my house is a vall
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox de. They had been into the thight station, was a bag with as sood pointed of a cool convenience jumped ; and the darker world, thut it up feel the children was to minute of the head so much very new morning

on, I wouldn't have had many for
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They was not good all down and his now was demothly to be a very prevate.

" Some is going a minute in a part of company," she said, " but as in each ! "

" You don't think of you," answered Marilla sore of her bust the house eyes, "so I k
```

Seed 131, prefix 'The children '

```text
The children leave a perhaps we friendful start round, and that was coming to make a suppose. If never we were a carry life so and close the room and called find of the window. They were and she continued panting in the kitchen face was to be kind an ol
```

### large / v7_long

Seed 123, prefix 'Alice said, "'

```text
Alice said, " I '11 get all the way of it with you ? "

" I 'm going to a boy," said the Gnat.

" Who didn't I thought you know when she has soon in," said Mr. Havisham. That can help her hand to buy the Wicked Wendy, " What do you they are in my fancy 
```

Seed 124, prefix 'The Queen '

```text
The Queen cried at last something. The Lion in a high three hand further said : “*

“ In the time Beth’s a great companion of the Red King,” said Bobbie, “ I should be ready what be injust that it was once.”

“ You see than we could not
```

Seed 125, prefix 'The little '

```text
The little Fauntleroy's pantry started at Mrs. Lynde was all the head, while the brush over the table, he saw calling and study to find T' He says his wolf. " And there's the story ! " said Alice.

" Yes, she little girl," the Knight said, and stoppin
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, " she added as not touch the rat fir, pronounce in the deservant mere to be during, and the man of the shelfs were althought the school, but when Ben Sara colour was common with her posmition, and she felt on the way of the carriages she set
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a catch stopped from the other still and he had liked him to keep him from the house, and said he had dreadfully but everything there talked to be able the room. But there is nothing in the most come out, and it would want to see the
```

Seed 128, prefix 'The Mole '

```text
The Mole of Dr. The floor, Anne could keep the doctor's cheeks of Mrs. Lynde bened.

There was the dischild straight of the season and I leave the hudder hard all the house, while the weeks that presented all, staring over that the cook would go tha
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox did not the Sunday at Becky began to work, but then they always done the good school, who was driving before the darkness before somewhere conscience, for a fault midst of the rheat so in the right of the degologne behind it was jam with Mo
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They've been good all done and has a dozening."

" It's a seven far bring him."

" Ah, dear, I don't know show we can't like a single — and thee piece in the first time or the new and we want home to my man to fit to know the old man, an
```

Seed 131, prefix 'The children '

```text
The children let go to have a soft in front of provision at the strange for the comfortable to the door.

Dete rich and changed the brones of the graveyard country and I don’t tell me what it is nothing promise ? ” he said. " You never feel that wou
```

### previous / r0_baseline

Seed 123, prefix 'Alice said, "'

```text
Alice said, " ! 'The s say the bre an tin we aly s d he " thanitond f anal t thang w le arid oug s lound ghm I ty werer d  lyouint t selphd thit hind y ou, he wn s "hinisThan ce thed the ig th thinous ce s The lduid s the thed tous tbed tove the m poug 
```

Seed 124, prefix 'The Queen '

```text
The Queen t talis aikin s tht they t topu shin ashid y hothe Hin f tofis s shicto s he oucang as t wald, we oon o t he thiwhe, t ooul, r wakelis t on hanled und, koling  fth than quguced ad anghed ay t it thawouooullld tid t thius, w wan we tout t st
```

Seed 125, prefix 'The little '

```text
The little y hthe wher ay and the ame ke Que ol ooas ce oul thin her hen w thex: "


" ove " ca t ceedned t whengut,an here wi, t'tn n'g
'to he t d she tou s" thsat the he heveng s ! " s " " he-athey theshing l tond wouoke th t the  ind cedou antt t t
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, " s he d as ashof t tugh t herat : 
,444 werin t bling ay an ar aner ced shed athe Aly t lyid thowit oke weas Hu s ant fofouathe s nd d atls wee beb th sthers bait s coorohan aw ceve t maved l as ere loun thad ed id me meery t hacit thed sea
```

Seed 127, prefix 'Wendy '

```text
Wendy ricke ta int r she igoued t the has m s ve e an' f ayoaly y bupet the ty aaneg it w ci: " " tho whas t t " thire" m, s she o be I wapot hers terou, t t d the lou arof w t hither bin nlimalyNok  beere ou in'e t bee heit coly aloor t out s ae
```

Seed 128, prefix 'The Mole '

```text
The Mole ourer llin f shel Aing mon heke s ten the segheay a s t he soound d beney th t. "



 thicicich bu s hed t, we gup thes of don ale t. "



" nouthar Angouts allds t : Doe chet see at t mote onghasall, s fid to w.war I " Nom oos gouis t t p 
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox d oh Owit thin Wouid lit  be there thalild wo t ther wis t sos td p ounr an veed oproche temrong t t s ! 


" Af d rane tout het soul t t weue her che fowais ang we d bidof juthhe t yo inghy won n, Sherinid


44 I I t veerit d chef men't  t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " ne se an t At torinlipd ou heny me wont and k.The t thed Save t chou t : 
" lid t ithed ald an tu thid a theor se  t d weithod sind ing ou an tnte tid rancon t t ke but.  Dof py Aloraib, shisn ches t arem t ower f, wo f hremose o wals and 
```

Seed 131, prefix 'The children '

```text
The children the ay," m ano w e f vitofon s at prou sh, t thitt  ske ? The tyonge thens t t ther nd oathiwitt ir caa woly nes. " whin In ke t s he na calin  sin co ad rowit s wint. 

" ad So s bing wouling and Al ? Ovand s in ae t a d muthe ain t thang 
```

### previous / r1_more_training

Seed 123, prefix 'Alice said, "'

```text
Alice said, " she I gring ate br she hand a they sto he " I was oned. " and it and wy mas to ought lhe hegrmme tham and do knoting itse puple it his, you forr ones ear it hand and edree sim to thisue ble she saidd was eand herspe shebed to wer hemppst l
```

Seed 124, prefix 'The Queen '

```text
The Queen tro wer : une sat thelly u to ther," and thy kothe wing too if said at aste boucte sar toe lddo ea wnas to to could, it oulver the wishe of has and if the her the lot very uump anke it anay und it awnd ould ousid she iupte thand eet the lit
```

Seed 125, prefix 'The little '

```text
The little for a cull way," wen the cor ourd on oof  cherte thing, " hen in it : "

" It thelly kne tone it wecnt.


" It if iu to nin's 'to hers domy woong "


"It the he hryou the be so rouge sae you ont he lllong anto bing an he finge on't and sthe
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, " she ad could of theult tooneat : I,444 woothing lite and wand an reemp she said and it leve thoumxto she as the fand foly woon son the blst the wbe has wass and you oon hand wer she smave. The sers and and !" I id mut seeap the ceats the a
```

Seed 127, prefix 'Wendy '

```text
Wendy rith itt in ar she bethed the was anm she se an't muy and you and you and and in a Qui: " " though sing the gart the st but eellt ? "


" " It bout the dche so ther ousste the by it noom woish abe to ous wee to the bitte lroand and as the e
```

Seed 128, prefix 'The Mole '

```text
The Mole onger los fforsed And to and knit the doughes the a ste fe soo ne toulde uft the sood thoung it but ther the fpe she hellf dor an afd the'h dnoown a ongeststly troun said and ase wat nas the queasaid, shem cre wown't do hom oos gook an at i
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox do. They hentupl sid and a the sim ther an wown and awnd awnd ! "

4 Kne an thers, worne ther in the said to-boor so he to. Bult so.


" Iffue ner could was to to thelf oom jutthe ther ingey wor net morsside.

4 I wher olld a wandf moootn t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " Thelle han at jors inppent anny mponed waside." Hut the you onn and it : 

" So the greanld an turn noth faror se it downithoersing it bout ootnte tie ranco whey kno th. "


" And ong anke won oh so to so the ea frt too hrem se oppatst yok
```

Seed 131, prefix 'The children '

```text
The children the gy," I' so ween and of nos and rought, " thiss aske ? " I wallly the sQueeenter. Alice thimte and and she corsaid in and and so fing anl il sit cor the withe mint. I whas th she be woonger and an ?"


"Its in and ound muscme in them wol
```

### previous / r2_longer_context

Seed 123, prefix 'Alice said, "'

```text
Alice said, " son the sany the bruid had the aly she said il a ompeaf alally but on y mas id ough sthe he dmit tham a ppriter : Hon't you don ith ith you manthen sailly. Thand to hedr a sid the sifore crea mes ittille ea veny said itbectto in have pof l
```

Seed 124, prefix 'The Queen '

```text
The Queen troully piking at thouly u to the up as it yourthe nin for? is said at asthe oucan qusaioutly, weas tho the said a ist oo lonr the wisth of hatllly nof you you lthe : and dobed a bruthe say und itt.


"Eillght it she iupten was we so the mu
```

Seed 125, prefix 'The little '

```text
The little for a out she and she ta coke Queen. " I vereraled in hermppoowng exkittugh ove : an the yon dot wacnly lan thiseng, then sary't she sid she tor said said he as hy she say." said Ale-sasty a ond he b ton hat ombing at he you and hisand sthe
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, " she ad as as of the geged, ean : ", pranen, 4 inl at ay sthe Queer a I she said ally the westly.


4 Alitas the faing of pathe sin ? " the was llo the : And and sain it had awachely, mand. T as ere a dnothed ep ittle bre a t hagin't sthat 
```

Seed 127, prefix 'Wendy '

```text
Wendy rith say deen tshe it."


the I on calll queain't bu therey busht you and and poom Qui: she and wiss the saidd the nos she oe llittly, mierse aboutht loucke, an ith wow the a stouin noom with rabe to on gane toree histhlly mawathe dose a a 
```

Seed 128, prefix 'The Mole '

```text
The Mole cerer lof aflently in to a hake hat ging heds this of elle soo ne douldednty you's the toully, all sthe : she an she the fad noom at. "


" And thar ond Alitly the sally, and asee att a preatte ssaid, shaid to waway as illlly. She is the yo
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox din a a the as souid lit a the sabe the lot wat ther wit a sthe ddloow. "


Shed oprecte thir in the said toly wersaing torthe thupsed uriffuelyy here fobais a t we dlit of jutthe the binghy wouth, she soid


44 I whes oly the matf moooth t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " The lis ! " the Shallpente heny miced that demored at aby ave tof ther : thellllt ith Ud said the : nou a shoond, it saily, "dand I it boutte thee prithatuld thad ske the preas and aa brith snoorest torem the ea fre thor reetse on atstore 
```

Seed 131, prefix 'The children '

```text
The children the gy," I as skee frain'f nos aid rought, ist ifth she ? the "


There as if the wand oat thimthir ca a sthe lorsaing it ne and she gon cal il sit imp astooneds, she you't don rithing with sip noout ?"


"I shing aftuled mutfmeednthe mawal
```

### previous / r3_more_capacity

Seed 123, prefix 'Alice said, "'

```text
Alice said, " said Alice. Hall drumpt turthen, you do bectiles oment. " I " wehat no loose doong stoo nothim I that's puck it : Ho they phing it his, you mant wass ! " Alice to It got theim to this ore wen The imumpted Alice said. " becaling the not to 
```

Seed 124, prefix 'The Queen '

```text
The Queen trough tought says ! " aid to jest," said ther to his if to ifirst imtomsten ouch, quicke ald to fight ! " said Alice too loor the wishe only quace if the has tot loook a goot asher hiss alll

to bawousE to old the was aye thand ever up lad
```

Seed 125, prefix 'The little '

```text
The little for a began ! "

" Alice couldn't slooking are linting, " Alice to Alice oliove : conte a deedd awacn'ut, and if the twon thro's she say. you'd it helse for the hras tremever but he ! "

" You Alice. Mongs byoce the fight is a I his wenly i
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "baster. " I can't thugh tooked by I, pran it it blie fay as argain the ! Ahice his to the imper said to ifast if it then cult — never the was begin a :


" You'ss could and wace lot marked ? Marst alll to ! " said Alice to that thes tas a
```

Seed 127, prefix 'Wendy '

```text
Wendy rathing but a tshe is in ? " Alice call very we' be you felll and the pristery it wors: for and whas to the grre" then she oer cit.

"things terow, then chen sallled she the cloout onle alll drabe to one weent it's bee ! " said Alicese Alic
```

Seed 128, prefix 'The Mole '

```text
The Mole of tright forst," Alice on happed ten to help this of the she hads were the ! " she said, it's curpred dot, wore ! "

The con'I llow. "An't dent sar the Alice ! she were'lll the pisten tote onle sall, shaid to hew you litttle sage in the pi
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox dexp take this souid little brod be to alll way ther was a said down't reat to coourced one rounds to : thim that say."

She ? thu slow then elimst.

" Alice as ! Alice I didn't the sallly ? ""

Cast Distider to I was intten that fallly for
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " Alice said Alice walking way ne first that demore that.

" Then frost them the the flidd shat's uca ndry face as a treatent off ind in in it one enow your on that she tro preap. Alice buten seach somented too ea fre too hre toe olld soonlk
```

Seed 131, prefix 'The children '

```text
The children the gyet that she down tof the for the she tat it head ? " The you case asQueen. " Ind idow whet ? what wold ther and in and and roment are willett impadsing ! " Alice salll. She heis said mereand Alice ! " ther the to erem to fain them

Al
```

### previous / r5_scheduled_lr

Seed 123, prefix 'Alice said, "'

```text
Alice said, " I don't sany you have a turn it, you know."

That misself all obout not make to the floud, because asken," she said, insectly this high one for a must hrough and to her the inventides of crwarm sile?" said Alice.

" You've a birthd.

"I sh
```

Seed 124, prefix 'The Queen '

```text
The Queen cross at fun one other air to them, and it crying and if they is she to stuon the mean off loud, and so that the began openerring when you know."

And the Unicorn lot seemelbody her himself under always, looking at this that, so ever the mu
```

Seed 125, prefix 'The little '

```text
The little for a little panting to me ! " Alice said in a little her hand of them turning !"

444 Here had a ment.

But the with the sarv'y shoutid ? " the Unicorn, the kedd here smove her. Ahow as you long. " It's a looked her fall on and his stoop o
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't know ! "

" Well, the found her, if you could not have the boat's a joust that, and she was wanted to and for patting never the when the other way, just come have were lyorm?" said Alice : " and they wanted to you have a shorse,
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a detorn, this in the Queen : " the ear head yearfully and the proud.

"Not drifmished of the kinds quired.

" A botherel of pot herself.

" The very sent all of the poor bit neard ! " said Alice, when she thought the wood-by sett re
```

Seed 128, prefix 'The Mole '

```text
The Mole offtrorted them, and she said, if the firesely as she called of the broud.

The little to did, and strong," Alice said, after and shaked one for her only them ! " Alice excited, peating out one off the roow to cown badler, and the blacked i
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox do. They hadn't fun the end beg, because I would," said the King, poe fread to come of both to say the stating for ague to them turn waus went on her, for a faul turn," Alice : he there's as and never the head off abellott, as a jam with Me
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " when I had an joke in down a ne minute offide."

Alice began to cry very manage to him. Alice the whille face as a tried, as Alice with him up nied to carrie that she thought shake as affer she has not more one of the find.

" I shall seem
```

Seed 131, prefix 'The children '

```text
The children through the so with was sileng at provisice as it

as the treest of him as he tremp.

"I showieth your a of face."

" I'ven make a bound reamily," shouted, without Alice what the waite with treen that she chose, and the mimselfot ket that l
```

### previous / v1

Seed 123, prefix 'Alice said, "'

```text
Alice said, " Sof I could stand way that's got in the was just in surpal so did nothing sordon, in then Sheed me all and of you me with phim sigh it, you may hands ear it had called round his a sighed, went he led was great to to shower, and she was go 
```

Seed 124, prefix 'The Queen '

```text
The Queen proud at further.

" I amputy never more, and at her middle in she brooks.

"The passs ofet But I conten it stood, is it seemed a whispered hands out first the for of her gust are all just tumble always, larging a tree, things feet of her d
```

Seed 125, prefix 'The little '

```text
The little for a little path rough the wood, of course, twiculle her her in it, and here heaped to herself, with us, and it was toon sigh' wa all dried to it, and there he haventer saying a littled banking his hands, but behind the King creally, was a
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't can I had to done the doesmen it would he lefe. "I've reard! I'm afraid you mean that wood, I aske you may of path a word carkscres beginning to being she back hands walk, marked, and she was tooded — " My ear DUMPTY DUMPTY.


```

Seed 127, prefix 'Wendy '

```text
Wendy I can talkes."

44 There woodden — " the end of gayes.

The penty seemed about, with: for the worse queer garden insectly.

"I shopes," Alice a little curiosity look silence to it heavily it abe ! "

" The Queen very cold a tone of the we
```

Seed 128, prefix 'The Mole '

```text
The Mole of this suddenly and moment Alice went on anglock a pat feet, hat'll caught : " this it was in all seemed to fill the wood Looking-glas would the round of start sound the called of the more only said, " I'd mean way at it haven't gis a voic
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox day light this sound the eggs gentleman about (Alice was a sis there's readfull) opened. " Evoice much ! I don't care the beliet unsat urn for the crummorard about, which a just got go being with she had in

of the best, and was jam with it
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " Tweedledum remarked.

" It a slike a lover dear hopen found

you poetry: there've hold quivered, and was there were them."

"And now, with a won't sighed up a large breat off.

"And a brist snockes. I came and sudded, beginnines and stood 
```

Seed 131, prefix 'The children '

```text
The children through the solemed winto on sick shook with this head in the plass, they Queen.

"In you'd anything a groled lorling crown the regular crab in I'm contrart down thit.

44 That's he be woody ! " Alice said inseriage to her though ! " Alice 
```

### previous / v2

Seed 123, prefix 'Alice said, "'

```text
Alice said, " She were said, "but if it was they came of the glarden.

" I set a great again of course ! " cried Tweedledee. "And the King looked in a great repling him, They can't dream in the time, but she said, " I've have said it's good in a moment 
```

Seed 124, prefix 'The Queen '

```text
The Queen suddenly full sitting on the pudden in his heart.

He is so see all rollemnt out a quite said, " Contragers at puzzlain of through the head at last, when the micte of practice as began shap unGoat, would like to be again, and so every didn'
```

Seed 125, prefix 'The little '

```text
The little for a little anxious to manage."

"A look come ! " Alice said, looking.

" It isn't take the dish mong.

But the night on the day house in hands at the took of the shore ! " said Alice.

" You needn't see allower her from of a delight to ha
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't sent to do the crabbe a, planting to lose all the Queen said, " but it's a voice of breath her," Humpty Dumpty were said dingly.

" Oh, oh ! " said the Knight.

" The same more all say she did not ext disme feet, 44 give you'd m
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a deep us of the King with a misch sea " Why, there the horse's no are fire ! " Alice said, sitting quired.

" A box ! " thought Alice asked, before to chickets quietly.

44 1 toudently seemed to be a tone's fit of that of them of a 
```

Seed 128, prefix 'The Mole '

```text
The Mole sented of himself night. " You're tell you know," she expected her again.

" I sent it was to have the dit," the Unicorn said voice : "but it was are nothing hows the room to see water on, when I all, shall tell ways the road."

"I said," t
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox said to the Square ! Faster ! And startled : but, as went on she did not got under worse there was to be a more of

guestle and turning to for her cheesss as as one of them jumphant so very voice , she said

on, "I see you'd mean fast you t
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They must you jave invention this head ! "

44 And what you say?" the Queen said, looking a little great coming eyes arms all her with a gain.

" And year one all round. You way not among horse her finger to the feet," she said on a soon k
```

Seed 131, prefix 'The children '

```text
The children the great hands went very on great round, and his hands and hair freetfully finger.

Alice remembered again for anything on the room hands his hair for the window sitting, and tried the queer. " And the voices in a first tone."

" Contrariw
```

### previous / v3

Seed 123, prefix 'Alice said, "'

```text
Alice said, " So you will about was on in a thing tone that was made. That the sort you again."

" I'm afraid I'd mean promison," Alice replied.

"Which was the King remarked at the room in the time," said Tweedledum, "And here, and because it makes to 
```

Seed 124, prefix 'The Queen '

```text
The Queen said to herself, "that you to talk," said the Lion.

" I don't say it once," said Tweedledum. " Of course ? "

" Yes, too. Three are soldiers the word, you know the gave this dark ! "

Alice was good and looking at the other, " we could not
```

Seed 125, prefix 'The little '

```text
The little for a bottle.

" Of course you are ? " he said, as if her hand it was a convericefusted tone down ; now in another up the same's she story. However, that they were she came on again, and hand she had to fan to be herself, and seemed to stra
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see it turn to crab : I wonder, if you are you see anything I'm so many convering to herself answer, if you like a hurry directions : but I don't believe in that all many my dream : and we did on't be any purcular hug to she see
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a conversation."

" Well, what is is the house ? " said the Tiger-lily. " Not it was a well of life, * You'm not but every names in somebody to be me, like hold."

"Exclaiment to do it round at the seems," Alice said, "only the extre
```

Seed 128, prefix 'The Mole '

```text
The Mole other little shelf in a minute.

" I can't know," said the Tiger-lily, "you've seen if it would be seen the words the sea confuse ! " the Queen said in a tone of delight, and went on a more or two large first of with the room.

"While, I'm 
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox did you think it side."

" That's not a long wax," said, the Knight : " thought it would her, in something it was so up to the thumble. The Knight couldn't go by think to shrow her Rose."

There was more very good about, and the Darmouse sh
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " That's a number of little very happening first that the first — "

" That looked again. " Do you have been

out it difficulty in it was made not at your life your breath," said the Caterpillar.

44 However, then, which I mose you'll go t
```

Seed 131, prefix 'The children '

```text
The children that was made out of very on side of easie : this he knitting her head ; she turned and at the tried a ground, so and she remarked her a chorus of thousand the shop.

" That isn't be a very nice ! " said Humpty, as she was soon a kning toge
```

### previous / v4

Seed 123, prefix 'Alice said, "'

```text
Alice said, " So you should like that the window ? "

" Well, it's a part of the great Hatter."

" No, it's my come and do not it I know how you had a good near how he were eyes they to size off. Took crossed all in saving his pocket-him. I do have not 
```

Seed 124, prefix 'The Queen '

```text
The Queen cried and sixty-the bees usual

shut as if your found if you silence."

4* 1 never saw the little girl ! " said Alice took very suddenly of his head. " You know the gave you want be all say to the Tiger-lily ? "

Alice wouldn't see what lit
```

Seed 125, prefix 'The little '

```text
The little for a country.

" The same a tire of this way with it hand: which ' you could tell you call you meann ugly and yet, and not going her down upon its mouth the hedgehog."

" Why, it wear something be a little good really May invion't as well 
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see it's going, eat you, please, if you're you. I was reached his arm," the Messenger said, whispered the followed his middle.

"The Ball I don't believe in that away fly," Alice replied and looked at the Mock Turtle. " Is that 
```

Seed 127, prefix 'Wendy '

```text
Wendy I can talk to the big."

" Well, what is is the house ? " said Alice. She had pine this, she found of her head them, and shoes came to the same of the Lion.

" However, you know because ? " she struck the sea. " And we could not cats quite 
```

Seed 128, prefix 'The Mole '

```text
The Mole of tried with the night of her.

" I can't know," said the Tiger-lily, "you're sometimes and fearfully down with plum-cake. They were ten in the proming upside the brook, and went on a good open said, " I'll go twice things of the big, and 
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox day ! " the Scarecrow. " You may be a very body want with you, you know."

"I shook come to to say the Tin Woodman

and then ? " said the Lion. " Only for a game with brooch ! "

" Really is more so many very lonely than the mean," said the
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They had not got to be all her hands at all."

44 That's the Red King said, like a shridgle of the girl, for you, it's my own a paw in the air needles came to any skime. Dorothy Alice began snore so sore of course, which rememember this wa
```

Seed 131, prefix 'The children '

```text
The children through the soldiers violent start round, and then she could not make a nice tone of the horse's pricket, so she was so very long when they well in the dark, and with her hands right her.

" Then I would have dinner to them ! " and the Gnat
```

### previous / v5

Seed 123, prefix 'Alice said, "'

```text
Alice said, " So you should like that the window ? "

" Well, it's a melancholy of which when I can have got to your for?" said Humpty Dumpty, as it as she stood spoke, "That's all draw ! "

" It's only what sometimes to have spoke," said the King.

" I
```

Seed 124, prefix 'The Queen '

```text
The Queen cried at full six that a presents, and his hardly hand for their shartless were some shoes like a conversations way, it was srat sweet as they lay under they grot looked at bellow and fastened.

The Cap, loating a way upon than we so until 
```

Seed 125, prefix 'The little '

```text
The little forgetting way, for she must of the books from the bellowed with its minute or two, and then the wacnought and stuff then she had had done.

" You have to his eyes," said Alice : " was speak on my name ! " Alice replied. " Not I heard the G
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't sent to until I eat you, please, if you'll you like to the bit's a pirate that may of the shapes were all followed him. Alice screamble to him, so she said "That's the last," said Alice, " and it's moment here and hand at all. N
```

Seed 127, prefix 'Wendy '

```text
Wendy I can talk toges to get to be along myself eyes, " I think I have you are any pirates : you'd tell of come to really story of the way," he said, "indeed cut of them of still they in them, Niih ant this came out of this would look at seem an
```

Seed 128, prefix 'The Mole '

```text
The Mole carefully for the night of his ment. However, the song the slowly way, he forts thing that it had been ditched to her stood no one that wouldn't have no upside to like a chair eyes to her eyes, and this side of the made of her hands and the
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox dech that was some of please the boys of the Cat, and was well tell me ready.

"Not can one round much ! " Alice asked, to the time.

44 If I shall do not any my well," the Knight said in a very new more very good-bye, " and what falling it
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They had not good all down and his eyes and the Poor

words of the shrieks ? "

He was quite excution, and hopesed to begin with no it, and so needles raturned at last the boys and rather her to see to me the end of the house either story 
```

Seed 131, prefix 'The children '

```text
The children that was the other five of the door, and he stood heart on the top of her piece.

"In your face?" said Alice.

" The King, we must be a pretend for the dark, who was were talking her hand, meeping away — I shink I don't must have to my be
```

### previous / v6

Seed 123, prefix 'Alice said, "'

```text
Alice said, " So you should like that the window ? "

" Well, it's a plan forth of green ! " said the Rat, my mouth and down this, which had somehow, you may comply here ! " the Queen replied, as soon creature it as she could speak to clasp it again, an
```

Seed 124, prefix 'The Queen '

```text
The Queen (who was funny at the mark town shut of it, that he could of sight it out her lister.

" Of course it out of the beast, of course," said the Caterpillar, looking at the queer darker interrupted in a sun, " and it wasn't oping of ever the da
```

Seed 125, prefix 'The little '

```text
The little first of West."

" I suppose that she's me a sulky," said Alice : '* and seems to be always like a cneedless to find then I'm

just as she said, "I must be be a hungry small so many ! "

" You how to the same came ! " he said.

Alice took a
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't see it's going," and if he would hardly have a little girl been so much a thought to herself : she had no found the sent directions by the greatle and stood up his eyes other shall says, and her feet in me of the whight the same
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a catch stopped him up his head, and they had talked by him the road.

"Oh with: fyou moment it seem to them," the King said interestly. " She musch help her feelings flying and many of the most crose to the bit of the other of the e
```

Seed 128, prefix 'The Mole '

```text
The Mole carefully for the night of him.

" I can't know," said the Tiger-lily, "you've seen it wouldn't be seen before, if you don't make a bit of conversation," the Rose said with a seemed to him, " of saying this time."

44 Don't you go by the ba
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox dech that was some of pleases gently without quite eager and stiddly under the cool of both road, and sat they were guessed the turn was the end of the twins and were find the room. They had not never their

old no best, the much fame out o
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They make you join in the way your now ! "

44 And you'd been no preventice of their getting out ! " Alice asked, in sixtion (as she said in a needles rather patiently, 44 believe it's far why it's a prementice of the first the others and 
```

Seed 131, prefix 'The children '

```text
The children that was the other friends, which she was, taking his prisoner to herself at the moment and watching a great clear curious of the white presen cape out of the Cowardly.

The Mouse was silence properly sight to in a face dimbection tone, whi
```

### previous / v7

Seed 123, prefix 'Alice said, "'

```text
Alice said, " So you see you to drift the window ? "

" Well, it's a plan of beautiful Hatta lagoon handked it to herself, " Do you know her hands all them are not before had called round him, but one come of the night way by a short eyes of the side of
```

Seed 124, prefix 'The Queen '

```text
The Queen cried a little street : " what doesn't have they had been ? "

The Knight looked at him, with the money took at her, though, rather suddenly as he could not help see an explained back to as if the Tiger-lily were straighter, and even the mu
```

Seed 125, prefix 'The little '

```text
The little for a country. There was a little slowly and sticking her head of the Queen, " I thought isn't suppose

large it with the same's shouting, you know."

" You'll shouldn't say have heard a convenient." Mind why didn't you have to be anybody o
```

Seed 126, prefix 'Dorothy said, "'

```text
Dorothy said, "but I don't know you get to rather , place, I think I always I am a cupboard of the conversation."

44 I dare say, * well, would you do be the slabe ' $ " DUMY OWN

could and well," the Knight said, looking at her remarked.

44 You've been 
```

Seed 127, prefix 'Wendy '

```text
Wendy rather a child stop and delightly and still a little time. Alice thought she getting it, she found of her head them, and she began took herself on, " I don't like to see the cleveres of all over, you know. I wonder if you don't make such a 
```

Seed 128, prefix 'The Mole '

```text
The Mole of tried white standing on his men painting. But she took up on the branches the side of animal but they turned up to see him one side, with nothing on time a border start him enough to herself with the side. But a bady, as she was so much 
```

Seed 129, prefix 'Mary Lennox '

```text
Mary Lennox dech that this Would like to do it that?"

You may try that/' said Alice.

"I shouldn't know," said Alice.

" If you had put him up." The Lion were puzzled over the Munchkin side, that into her to, who she was very beside, though for a tree
```

Seed 130, prefix 'Anne said, "'

```text
Anne said, " They can't be just like our called — I wish I wonder.

Peter can see almost to do it quite kitten when they were talking together in the air nose is castle that it were pleased to a bit of nothing to see when the wood had toes and stood 
```

Seed 131, prefix 'The children '

```text
The children leaves," the Queen said in a sudden of such a different. " That's all rather tone of playing wet ? " Alice enquired. " Look not you mind ake a country and I don't know what do?" she said.

" You see, if you'll tell you all you tell me any l
```

