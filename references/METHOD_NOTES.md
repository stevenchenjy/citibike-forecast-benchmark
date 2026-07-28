# Method Notes

- Forecast intervals are 60 minutes.
- Pickups and returns are modeled as separate nonnegative count targets.
- The main experiment excludes weather.
- Forecast origins and folds are chronological; all learned transformations are fit using training data only.
- Truck-routing optimization is explicitly outside this benchmark version.

Detailed data mappings and inventory ordering conventions will be added only after source inspection.

## Source mapping and time policy

The audited VP-RNN 60-minute files map `count_pickup` to pickups and `count_return` to returns; raw `inventory_change == -1` and `== +1` respectively validate this mapping within 2018. Capacity is joined from `station_information_citibike.capacity`. The core station set is exactly the 30 source demand-file IDs; the smoke set is the first five numeric source IDs (`128`, `151`, `168`, `229`, `285`), chosen deterministically without observing demand values.

The source has 24 unzoned civil-hour labels on both 2018 DST transition days. The canonical panel is an actual `America/New_York` grid with 8,760 hourly instants per station: the source's all-zero nonexistent spring-forward 02:00 label is removed; its ambiguous fall-back 01:00 count is mapped to the first occurrence and an all-zero synthetic second occurrence is added. Both fall occurrences are `data_complete=false`. Downstream fitting, validation, test scoring, and decisions must exclude incomplete targets; the source counts are retained solely to keep the canonical grid explicit and auditable.
