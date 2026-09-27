# Survey of every stored circle packing across all SI-v2 runs (aws-big, jason-mbp, canopus)

Scanned 2026-09-27 12:57 UTC. Every `csqv<n>.pck` and every JSON packing under the `results/` trees of all three machines (live and archived batches, held-out sinks, re-score dirs, v6/record-chase workspaces) was re-validated from its coordinates; only **strictly feasible** packings (wall and pairwise slack >= 0) are ranked. Records: Packomania csqv fetched 2026-09-27 12:40 UTC.

Files scanned per machine: aws: {'pck': 32682, 'json': 167, 'bad': 1632, 'skipped': 28525}; jason-mbp: {'pck': 31550, 'json': 41, 'bad': 1632, 'skipped': 27610}; canopus: {'pck': 2489, 'json': 0, 'bad': 0, 'skipped': 2341}

## Stored packings that beat the current table (6 sizes)

| N | best stored Σr | record | margin | strict beaters (files) | tolerance-only beaters | distinct runs beating | where the best came from |
|---:|---|---|---:|---:|---:|---:|---|
| 79 | 4.666630641409 | 4.666466802554 | +1.64e-04 | 4 | 0 | 2 | `results/circle-packing/2026-08-17_circle-6v-x5-20it30m-b/overfit-sm-1/bench/packs/csqv79.pck` (aws) |
|  |  |  |  |  |  |  | runs: `2026-08-17_circle-6v-x5-20it30m-b/overfit-sm-1`, `sota/si-v2-20260817` |
| 83 | 4.786683838728 | 4.786393807284 | +2.90e-04 | 2 | 0 | 1 | `results/circle-packing/2026-08-23_circle-plain-b/explore-plain-1/bench/packs/csqv83.pck` (aws) |
|  |  |  |  |  |  |  | runs: `2026-08-23_circle-plain-b/explore-plain-1` |
| 85 | 4.848035445604 | 4.847351780582 | +6.84e-04 | 4 | 0 | 2 | `results/circle-packing/2026-08-19_circle-bulletcount-x10-12it30m-b/9bullets-5/bench/packs/csqv85.pck` (aws) |
|  |  |  |  |  |  |  | runs: `2026-08-19_circle-bulletcount-x10-12it30m-b/9bullets-5`, `2026-08-19_circle-bulletcount-x10-12it30m/anchor-4` |
| 89 | 4.967596361015 | 4.966850487024 | +7.46e-04 | 4 | 0 | 2 | `results/circle-packing/2026-08-23_circle-plain-b/explore-plain-1/bench/packs/csqv89.pck` (aws) |
|  |  |  |  |  |  |  | runs: `2026-08-23_circle-plain-b/explore-plain-1`, `2026-09-15_transcendence-v16-oldbrief-x10/transcendence-v16-oldbrief-1` |
| 99 | 5.236809589677 | 5.236244728577 | +5.65e-04 | 2 | 0 | 1 | `results/circle-packing/2026-09-15_circle-6arm-v14-probefix-x10/none-3/bench/packs/csqv99.pck` (aws) |
|  |  |  |  |  |  |  | runs: `2026-09-15_circle-6arm-v14-probefix-x10/none-3` |
| 100 | 5.262884909339 | 5.262787458777 | +9.75e-05 | 2 | 0 | 2 | `results/circle-packing/_rescore60/circle_2v_x15_20it30m_20260817/explore-11/heldout_packs_even/csqv100.pck` (aws) |
|  |  |  |  |  |  |  | runs: `2026-08-17_circle-2v-x15-20it30m/explore-11`, `circle_2v_x15_20it30m_20260817/explore-11` |

## Sizes where only tolerance-riding packings exceed the table (not publishable): 28, 29, 30, 31, 32, 33, 35, 40, 43, 55, 71, 90, 97

## Sizes where nothing stored reaches the table (17): 62 (-5.2e-04), 63 (-5.7e-04), 64 (-9.5e-04), 66 (-9.6e-04), 68 (-5.0e-04), 77 (-1.6e-03), 78 (-9.0e-04), 82 (-1.4e-03), 84 (-1.0e-04), 86 (-5.6e-05), 92 (-9.5e-04), 93 (-1.3e-03), 94 (-1.3e-04), 95 (-9.9e-04), 96 (-1.3e-03), 97 (-2.0e-04), 98 (-2.4e-05)

## Stored packings better than what the repository currently publishes: 69, 71

These are copied into the sweep's warm baseline; the next automated publish exports them (source `census`).
