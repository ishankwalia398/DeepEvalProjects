# Bug Triage – Interview Notes

Notes for presenting this project in the "Quality and Evals Leader" interview. Project details are in [README.md](README.md).

> I do not trust an LLM judge until I have measured it against human labels, and I compare versions with data.

## How I would explain this

- Accuracy alone hides the costly error, so I report severity misses separately. v3 is the most accurate variant, and the comparison still refuses to ship it because it adds one.
- Citations are checked by code, not by a judge, because exact-match facts should not depend on an LLM's opinion.
- I measured the judge against human labels and found its bias (length and position). I fixed the prompt and showed weighted kappa improving from 0.27 to 0.84.
- With only 50 items I use confidence intervals, and I say "inconclusive" when appropriate.

## Mapping to airline operations

The same harness applies to **disruption-incident classification**. Replace bug reports with incident reports (crew, weather, maintenance or ATC messages) and logs with operational event feeds.

| Bug triage | Airline operations |
|---|---|
| severity | disruption severity (e.g. delay band, cancellation, safety-relevant) |
| component | owning team (ops control, maintenance, crew scheduling, ground handling) |
| root cause | disruption cause (weather, technical, crew, ATC, passenger, unknown) |
| severity miss | a high-impact disruption under-classified as routine |
| citation check | every cited event must exist verbatim in the event feed |
| judge calibration | the same kappa, position and length bias tests against ops-controller labels |
