# ACK: PI directives 2026-05-25 11:00 — Agent reply

**Time**: 09:38 UTC 2026-05-25 (received PI directives at ~11:00 UTC = 09:00 our wall clock; small wall-clock drift)

## Acknowledgments

§1 acknowledged. Continuing autonomously.
§2 understood — targets ≥5 mistral PI by 11:37, ≥1 exp_v1 trained + ≥1 eval triggered.
§3 EXECUTED — `exp_v1_eval.py --watch` is RUNNING (PID 2365272, GPU 3-7, max_parallel=1, poll=60s). It will fire merge+vllm eval the moment any cell's `summary.json` appears.
§4 understood — OLD vs NEW table will be in next push commit body.
§5 understood — multi-seed / other-model / forgetting-eval all deferred.
§6 understood — file-based comm, will create `analysis/COMM_AGENT_TO_PI_*.md` for new threads.
§7 understood — **No HF backend**. Patched `p0_reeval_orchestrator.py` (commit pending) to DEFER hybrid Qwen3.5 cells. Will document them as "vllm-unsupported, deferred" in scoreboard.

## Conflict flagged: §7 "No HF backend" vs current running orchestrator

The currently-running `p0_reeval_orchestrator` (PID 2336570, started 07:37 UTC) has the OLD code where hybrid Qwen3.5 cells use HF backend. Hybrid cells are alphabetically last in the queue, so they won't be hit for **several more hours**. By 11:37 push, queue position will be roughly:
- mistral (5 cells)  ← processing now
- olmo2-7b (~3 cells)
- qwen25-7b (~5 cells, PI)
- qwen3-1p7b (~5 cells)
- qwen3-14b (~5 cells)
- qwen3-8b (~10 cells, PI)
- qwen35-* (12 cells)  ← HYBRID, will be skipped after restart
- r1-distill-7b (~3 cells)

I'll restart orchestrator after PI cells complete (after 4hr push) so hybrid skip takes effect cleanly.

## Status snapshot at PI directives receipt (09:38 UTC)
| Item | State | Target by 11:37 UTC |
|---|---|---|
| P0 reeval done | 9/67 | ≥ 14/67 (5 PI added) |
| Mistral PI done | 0/8 | ≥ 5 (PI hard requirement A) |
| exp_v1 trained | 0/7 | ≥ 1 (PI hard requirement B) |
| exp_v1 evaluated | 0/7 | ≥ 1 triggered |
| GPU failures so far | 0 | (will document any) |

## Risk: exp_v1 train ETA vs 11:37 push
qwen3-8b/3000-step training ~3hr. Started 07:37. ETA 1st cell ≈ 10:30-10:45 UTC. Should JUST barely meet PI's "≥1 trained, ≥1 eval triggered" by 11:37 push (eval kicks in within 60s of summary.json via watcher).

If 11:37 push happens before any train finishes, will commit honestly with `0/7 trained` and explanation.

## Files I'm pushing this commit
- analysis/COMM_AGENT_TO_PI_2026-05-25_0938.md (this file)
- analysis/SESSION_NOTES.md (updated)
- scripts/p0_reeval_orchestrator.py (patched: DEFER hybrid)
- (no results yet — those will be in 11:37 commit)

## Reminder for 11:37 push commit body template
```
ACK: PI directives 2026-05-25 11:00

P0 re-eval: N/67 done; exp_v1: M/7 trained, K/7 evaluated

PI cells (mistral-7b) OLD vs NEW gsm8k:
| cell                                              | OLD strict | NEW strict | Δ      |
|---------------------------------------------------|-----------:|-----------:|-------:|
| mistral-7b/metamathqa-10k/relora_baseline         | 36.47      | XX.XX      | ±X.XX  |
| mistral-7b/metamathqa-10k/relora_diag_gated_S3pos | 36.47      | XX.XX      | ±X.XX  |
| ...                                                                                      |

[Critical flag if S3pos == random_drop bit-identical: ...]

Hybrid Qwen3.5 cells: deferred per §7 (no-HF rule). 12 cells.

Issues encountered: [list logs/p0_reeval/<name>.eval.log paths if any FAIL]
```
