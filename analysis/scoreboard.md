# Phase D scoreboard (live)

**Backend:** qwen3-* (dense) → vLLM-on-merged | qwen35-* (hybrid) → HF+PEFT (bs=size-aware)

Total cells with eval done: **46**

## Wave 1 (Qwen3 / Qwen3.5 ladder, tulu3-sft)

| model | method | gsm8k_strict | gsm8k_flex | hellaswag | arc_challenge |
|---|---|---:|---:|---:|---:|
| qwen35-0p8b | lora_vanilla | 0.2320 | 0.1713 | 0.5450 | 0.4497 |
| qwen35-0p8b | relora_baseline | 0.2881 | 0.2889 | 0.5278 | 0.4394 |
| qwen35-0p8b | relora_diag_gated_S3pos | 0.3116 | 0.3124 | 0.5273 | 0.4403 |
| qwen35-0p8b | dora | 0.3108 | 0.3124 | 0.5375 | 0.4505 |
| qwen35-0p8b | cola | 0.2972 | 0.2972 | 0.5287 | 0.4403 |
| qwen3-1p7b | lora_vanilla | 0.4147 | 0.4382 | 0.6519 | 0.5162 |
| qwen3-1p7b | relora_baseline | 0.5519 | 0.5565 | 0.6281 | 0.5247 |
| qwen3-1p7b | relora_diag_gated_S3pos | 0.3609 | 0.5489 | 0.6194 | 0.5358 |
| qwen3-1p7b | dora | 0.5201 | 0.5262 | 0.6314 | 0.5205 |
| qwen3-1p7b | cola | 0.5595 | 0.5656 | 0.6276 | 0.5230 |
| qwen35-2b | lora_vanilla | 0.3965 | 0.3980 | 0.6647 | 0.5060 |
| qwen35-2b | relora_baseline | 0.5436 | 0.5421 | 0.6477 | 0.5282 |
| qwen35-2b | relora_diag_gated_S3pos | 0.5360 | 0.5360 | 0.6484 | 0.5256 |
| qwen35-2b | dora | 0.5337 | 0.5345 | 0.6607 | 0.5324 |
| qwen35-2b | cola | 0.5125 | 0.5140 | 0.6552 | 0.5213 |
| qwen35-4b | lora_vanilla | 0.6626 | 0.6596 | 0.7621 | 0.6152 |
| qwen35-4b | relora_baseline | 0.7165 | 0.7172 | 0.7667 | 0.6553 |
| qwen35-4b | dora | 0.7339 | 0.7346 | 0.7701 | 0.6442 |
| qwen35-4b | cola | 0.7400 | 0.5504 | 0.7683 | 0.6510 |
| qwen3-4b | lora_vanilla | 0.6308 | 0.6831 | 0.7312 | 0.6118 |
| qwen3-4b | relora_baseline | 0.7566 | 0.7703 | 0.7288 | 0.6382 |
| qwen3-4b | relora_diag_gated_S3pos | 0.6778 | 0.7688 | 0.7266 | 0.6348 |
| qwen3-4b | dora | 0.4519 | 0.7824 | 0.7351 | 0.6365 |
| qwen3-4b | cola | 0.6679 | 0.7513 | 0.7274 | 0.6365 |
| qwen35-9b | lora_vanilla | 0.7680 | 0.7521 | 0.8014 | 0.6305 |
| qwen35-9b | relora_baseline | 0.8271 | 0.8294 | 0.8124 | 0.6911 |
| qwen35-9b | dora | 0.8143 | 0.8158 | 0.8162 | 0.6834 |
| qwen35-9b | cola | 0.8188 | 0.8241 | 0.8110 | 0.6869 |
| qwen3-14b | lora_vanilla | 0.7726 | 0.8014 | 0.8068 | 0.6246 |

## Legacy / earlier waves

| model | method | gsm8k_strict | gsm8k_flex | hellaswag | arc_challenge |
|---|---|---:|---:|---:|---:|
| llama3-8b | relora_baseline | 0.4337 | 0.4329 | 0.8185 | 0.6084 |
| llama3-8b | relora_diag_gated_S3pos | 0.4602 | 0.4625 | 0.8266 | 0.5964 |
| llama3-8b | cola | 0.4564 | 0.4632 | 0.8197 | 0.5922 |
| llama3-8b | relora_random_drop | 0.4594 | 0.4610 | 0.8176 | 0.5973 |
| olmo2-7b | relora_baseline | 0.6338 | 0.6338 | 0.8193 | 0.6195 |
| olmo2-7b | relora_diag_gated_S3pos | 0.6399 | 0.6391 | 0.8188 | 0.6152 |
| olmo2-7b | cola | 0.6444 | 0.6459 | 0.8184 | 0.6186 |
| olmo2-7b | relora_random_drop | 0.6459 | 0.6482 | 0.8192 | 0.6177 |
| qwen3-8b | relora_baseline | 0.7885 | 0.8006 | 0.7756 | 0.6689 |
| qwen3-8b | relora_diag_gated_S3pos | 0.8650 | 0.8704 | 0.7707 | 0.6732 |
| qwen3-8b | relora_random_drop | 0.8605 | 0.8643 | 0.7714 | 0.6724 |
| qwen3-8b | relora_diag_gated_S3neg | 0.8681 | 0.8688 | 0.7782 | 0.6715 |
| qwen3-8b | relora_train_gated | 0.8666 | 0.8696 | 0.7680 | 0.6672 |
| r1-distill-7b | relora_baseline | 0.7642 | 0.7642 | 0.6319 | 0.5324 |
| r1-distill-7b | relora_diag_gated_S3pos | 0.7225 | 0.7066 | 0.6213 | 0.5273 |
| r1-distill-7b | cola | 0.7612 | 0.7635 | 0.6336 | 0.5384 |
| r1-distill-7b | relora_random_drop | 0.7619 | 0.7657 | 0.6322 | 0.5350 |

## Coverage by model (Wave 1)

- qwen35-0p8b: 5/5
- qwen3-1p7b: 5/5
- qwen35-2b: 5/5
- qwen35-4b: 4/5
- qwen3-4b: 5/5
- qwen35-9b: 4/5
- qwen3-14b: 1/5
