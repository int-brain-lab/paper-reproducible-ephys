## `fig_data_quality`

Parquet files containing cluster and channel information from the Pykilosort 1.4.7 run of the RE, Allen, and Steinmetz datasets are in the `tables/` directory. 
This also serves as its own module (`fig_data_quality.tables`) containing a light loading API. 

- `clusters_{X}.pqt` contains a multi-index table, with all clusters and single unit metrics for each insertion in dataset `X`. 
- `channels_{X}.pqt` contains a multi-index table, with all channels and their depths and histology assignments (if available) for dataset `X`.

The `double_blind_results.csv` file contains the scores given by raters in the raw data quality exercise. 

The `plot` module contains some helpers for plotting the figures. 

The `scripts/` directory preserves some plotting scripts that were used along the way during the data quality analysis. 
