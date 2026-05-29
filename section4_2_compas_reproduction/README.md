# Section 4.2 COMPAS Reproduction

This folder contains an audit-style reproduction of Section 4.2 from
`Everything is Relative: Understanding Fairness with Optimal Transport`.

Open `report.html` for the full static report.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 reproduce_compas_section4_2.py
python3 build_html_report.py
```

## Main Outputs

- `report.html`: static HTML report.
- `outputs/figure3_ab_style.svg`: Figure 3A/3B-style alluvial reconstruction.
- `outputs/comparison_with_paper.csv`: paper numbers vs reconstructed numbers.
- `REPORT.md`: Chinese reproduction notes.
- `EQUAL_OPPORTUNITY_CLASSIFIER_NOTES.md`: notes on the equal opportunity classifier.

## Data

The script downloads `compas-scores-two-years.csv` from ProPublica:
https://github.com/propublica/compas-analysis

The raw CSV is not included in this upload package by default. The local copy
will be stored under `data/` after running the reproduction script.
