# Benchmarks

Two things the main README cannot show, because its figures come from synthetic
cohorts: whether the statistics behave as claimed, and whether the tool says
anything true about real patients.

| Benchmark | Question | Data |
|---|---|---|
| [`phenopacket_store/`](phenopacket_store/) | Does any of this hold on real, published rare-disease cases? | 10,377 GA4GH phenopackets, Phenopacket Store 0.1.27 |
| [`permutation_calibration.py`](permutation_calibration.py) | Does the permutation adjustment in `explain_groups` control the error rate it claims to? | simulation |

Corpora are downloaded on demand into `~/.cache/phenotopo` and are **never** committed
here — only the derived tables and figures are.

---

## Permutation calibration

`explain_groups` claims to control the family-wise error rate over HPO terms that are
not independent, because propagation ties every term to its ancestors. Simulated,
200 synthetic cohorts of 200 patients, 53 terms tested on average, α = 0.05:

| Procedure | At least one false term (null cohorts) | Planted 40 pp term recovered |
|---|---:|---:|
| Uncorrected | 75.0 % | — |
| Benjamini–Hochberg | 3.5 % | 81.8 % |
| **Max-statistic permutation** | **5.0 %** | **87.9 %** |

The permutation procedure lands exactly on the nominal 5 % and is *more* powerful
than BH here, not less — the max-statistic null absorbs the ancestor correlation that
makes BH conservative on ontology data. Uncorrected testing raises a false term in
three cohorts out of four.

(BH controls the false discovery rate rather than the family-wise rate, so its 3.5 %
is not a failure — the comparison is about what each guarantee buys.)

## Real published cases

Full write-up and figures: [`phenopacket_store/README.md`](phenopacket_store/README.md).
Headline, on 2,691 published cases across the 16 diseases with n ≥ 80:

- **Phenotype recovers the reported diagnosis in 93.8 %** of cases (17.2 % majority baseline).
- **16 of 16 diseases are robustly cohesive**; 104 pairs are robustly separated; exactly
  **one pair blends** — two chromatin-related neurodevelopmental syndromes.
- Across all 16 diseases annotation-confound risk is **LOW**; for the single most
  unequally annotated pair it is **HIGH** — the check fires where it should and stays
  quiet where it should.
- Using explicitly **excluded** phenotypes raises diagnosis recovery from 93.8 % to **96.0 %**.

## Running them

```bash
pip install "phenotopo[all]"
phenotopo ontology install hp
python benchmarks/permutation_calibration.py            # ~1 min
python benchmarks/phenopacket_store/run_case_studies.py # ~10 min, downloads 19 MB once
```
