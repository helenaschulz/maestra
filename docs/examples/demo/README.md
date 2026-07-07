# Demo dataset

`demand.csv` is the raw training data from Kaggle's
[Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand) competition (hourly rental
counts, `count` is the target) — unmodified, including its known target leak (`casual` +
`registered` sum exactly to `count`, and are absent from the real competition test set). Kept
deliberately raw: the leak and the temporal structure are exactly what the P2b demo asks Maestra
to catch live, not something pre-cleaned for the recording. Already used (post-cleaning) as the
K1 case study — see `docs/RESULTS.md`'s bike-sharing section and
[docs/examples/reports/bike-sharing.html](../reports/bike-sharing.html) for the full run.
