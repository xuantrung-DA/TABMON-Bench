# Contributing to TabMon-Bench

TabMon-Bench is a research benchmark. Changes must preserve its scientific
estimands, label-isolation boundary, paired-stream design, and audit trail.

## Before opening a change

1. Describe the monitor capability and its declared endpoint.
2. State which inputs are available at deployment time.
3. Identify whether the change affects a frozen schema or introduces a new
   schema version.
4. Add or update tests before producing benchmark results.

Do not overwrite completed benchmark artifacts. New protocol behavior should
write to a new output directory and record a new schema or implementation
version.

## Label isolation

Monitor inference code must not receive target labels, oracle losses,
failure labels, or intervention ground truth. These values may be joined only
inside the offline evaluation layer after monitor predictions or alarms have
been persisted. Observable meta-monitor features must be added to the explicit
allowlist with provenance tests.

## Endpoint discipline

- Compare methods directly only when they estimate the same quantity.
- Keep classification error, log loss, drift, attribution, and sequential
  event targets in separate result fields.
- Record unsupported capabilities as not applicable.
- Do not silently relax feasibility criteria, alarm levels, or failure
  thresholds.

## Tests and audits

Run the complete local suite:

```bash
python -m compileall -q src experiments tests
python -m unittest discover -s tests -v
```

Every experiment runner must have a matching fail-closed audit. A complete
artifact should record row counts, schema version, environment versions,
random seeds, endpoint definitions, and SHA-256 input provenance. Long-running
changes should first pass the smallest supported smoke test.

## Documentation

Update the README and the relevant workflow document when changing commands,
inputs, result schemas, or interpretation. Report negative results and
abstentions explicitly. Avoid claims that exceed the number of independent
datasets or the assumptions of the evaluated method.

## Data and generated artifacts

Do not commit raw datasets, processed data, fitted models, stream caches, result
directories, or notebook output archives. Dataset licenses and access terms
remain those of the upstream providers.
