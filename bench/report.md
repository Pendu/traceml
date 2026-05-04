# v2 benchmark matrix

Per (workload, mode), aggregated across trials. Overhead % and
peak deltas are derived by joining baseline vs traceml_run rows.

| Workload | Mode | N | Wall (s) | Overhead % | Step (ms) | Peak GPU (GB) | Peak RSS (GB) |
|---|---|---|---|---|---|---|---|
| BERT fine-tune (ag_news) | baseline | 5 | 318.64 ±1.0526 | — | — | 8.7167 | 2.0558 |
| BERT fine-tune (ag_news) | traceml_run | 5 | 327.62 ±0.9338 | +2.82% | 387.5059 ±0.2721 | 8.7167 (+0 MB) | 11.0809 (+9025 MB) |
| BERT-base FSDP (ag_news) | baseline | 5 | 61.38 ±0.1924 | — | — | 3.6479 | 3.3211 |
| BERT-base FSDP (ag_news) | traceml_run | 5 | 67.34 ±0.9397 | +9.71% | 200.4708 ±0.6257 | 3.6479 (+0 MB) | 11.6239 (+8303 MB) |
| GPT-2 small (wikitext-2) | baseline | 5 | 31.8 ±0.1581 | — | — | 6.3867 | 1.3466 |
| GPT-2 small (wikitext-2) | traceml_run | 5 | 40.62 ±0.9834 | +27.74% | 158.8878 ±0.1124 | 6.3867 (+0 MB) | 10.3592 (+9013 MB) |
| HF Trainer integration | baseline | 5 | 11.0 ±0.1414 | — | — | 0.8628 | 1.5898 |
| HF Trainer integration | traceml_run | 5 | 19.28 ±0.0837 | +75.27% | 2.794 ±0.0983 | 1.2487 (+386 MB) | 10.0535 (+8464 MB) |
| ResNet-50 (cifar100) | baseline | 5 | 15.7 ±0.1581 | — | — | 4.3567 | 1.8201 |
| ResNet-50 (cifar100) | traceml_run | 5 | 23.04 ±1.0968 | +46.75% | 278.8192 ±3.6418 | 4.3567 (+0 MB) | 10.9357 (+9116 MB) |
| Tiny MLP synthetic | traceml_run | 5 | 34.7 ±0.0707 | — | 55.7523 ±0.0776 | 0.9195 (—) | 9.9823 (—) |
| Tiny MLP DDP | baseline | 5 | 8.28 ±0.1095 | — | — | 0.8817 | 2.7705 |
| Tiny MLP DDP | traceml_run | 5 | 14.14 ±1.0597 | +70.77% | 3.8082 ±0.2598 | 0.8817 (+0 MB) | 11.1674 (+8397 MB) |
| ViT-base (cifar10) | baseline | 5 | 17.24 ±0.2881 | — | — | 4.1155 | 1.945 |
| ViT-base (cifar10) | traceml_run | 5 | 28.62 ±0.0447 | +66.01% | 258.0773 ±1.5652 | 4.5098 (+394 MB) | 10.5962 (+8651 MB) |
