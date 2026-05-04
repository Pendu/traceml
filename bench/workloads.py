"""Registry of v2 benchmark workloads.

Single source of truth for the 8 workloads from the v2 doc
(design/TraceML_Benchmarking_Workflow_v2.md). Consumed by
run_v2_matrix.py and build_overhead_matrix.py.

Schema per workload:
    script        path to the fixture (relative to repo root)
    nproc         processes for torchrun (1 = single-rank)
    tier          informational target hardware (per v2 doc)
    label         human-friendly name for reports
    extra_env     env vars to set when running this workload
                  (e.g. SIGSEGV workaround for fsdp_bert)
"""

WORKLOADS = {
    "bert_agnews": {
        "script": "examples/advanced/bert_gradient_accum.py",
        "nproc": 1,
        "tier": "1xA100",
        "label": "BERT fine-tune (ag_news)",
    },
    "vit_cifar10": {
        "script": "examples/advanced/huggingface_vision_vit.py",
        "nproc": 1,
        "tier": "1xA100",
        "label": "ViT-base (cifar10)",
    },
    "gpt2_wikitext": {
        "script": "examples/advanced/gpt2_small_synthetic.py",
        "nproc": 1,
        "tier": "1xA100",
        "label": "GPT-2 small (wikitext-2)",
    },
    "resnet50_cifar100": {
        "script": "examples/advanced/resnet50_cifar.py",
        "nproc": 1,
        "tier": "1xA100",
        "label": "ResNet-50 (cifar100)",
    },
    "bert_fsdp_agnews": {
        "script": "examples/advanced/fsdp_bert_minimal.py",
        "nproc": 2,
        "tier": "2xA100_SXM",
        "label": "BERT-base FSDP (ag_news)",
        # Known FSDP+dynamic-padding SIGSEGV — see fixture docstring.
        # Production runs must investigate before unsetting this.
        "extra_env": {"BERT_FSDP_USE_SYNTHETIC": "1"},
    },
    "tiny_mlp_ddp": {
        "script": "examples/ddp_minimal.py",
        "nproc": 2,
        "tier": "2xA100_SXM",
        "label": "Tiny MLP DDP",
    },
    "tiny_mlp": {
        "script": "examples/pytorch_minimal.py",
        "nproc": 1,
        "tier": "1xL4",
        "label": "Tiny MLP synthetic",
    },
    "hf_trainer": {
        "script": "examples/huggingface_trainer_minimal.py",
        "nproc": 1,
        "tier": "1xA100",
        "label": "HF Trainer integration",
    },
}
