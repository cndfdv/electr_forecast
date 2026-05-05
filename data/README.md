# Data

Raw datasets are intentionally not committed.

Expected raw files:

- `data/raw/ECL.csv` or `data/raw/electricity.txt`
- `data/raw/ETTh1.csv`

Use:

```bash
python -m src.data.download --dataset etth1
python -m src.data.download --dataset ecl
```

The ECL source repository stores the original semicolon-delimited electricity file. Preprocessing converts it to an hourly aggregate target for the main benchmark.

