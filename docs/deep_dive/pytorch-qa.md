# TraceML — PyTorch Q&A

A Q&A log on PyTorch — fundamentals through internals — covering the surface area TraceML's instrumentation depends on.

**Companion file:** [learning-qa.md](learning-qa.md) — covers OS basics, networking, Python internals, CUDA, distributed training, TraceML architecture (Q1–Q15).
**Walkthroughs:** [code-walkthroughs.md](code-walkthroughs.md) — file-by-file readings (W1+).

---

## How this file is organized

Question numbers use the **P-prefix** (P1, P2, …) so they don't collide with Q-numbers in the conceptual Q&A or W-numbers in the walkthroughs file. Cross-references between files use clickable Markdown links.

Questions are grouped into 11 sub-topics, ordered roughly basics → internals → advanced. Each P-entry follows the same format as the Q-entries: short answer (2–4 sentences), long answer (substantive walkthrough with code refs and ML/optimization analogies), and a "Concepts introduced" trailer.

Sub-topics (more questions can be added to any of them):

1. Tensor fundamentals (P1–P6)
2. nn.Module mechanics (P7–P12)
3. Autograd (P13–P19)
4. Optimizers (P20–P23)
5. DataLoader (P24–P29)
6. CUDA dispatch path & memory model (P30–P34)
7. Mixed precision (AMP) (P35–P37)
8. Distributed (DDP / FSDP) (P38–P42)
9. Eager vs graph mode (P43–P45)
10. Checkpointing & state (P46–P47)
11. PyTorch internals relevant to TraceML (P48–P52)

---

## Table of Contents

### Tensor fundamentals
- [P1: What is a torch.Tensor actually made of (storage, strides, dtype, device, view vs copy)?](#p1-what-is-a-torchtensor-actually-made-of-storage-strides-dtype-device-view-vs-copy)
- [P2: What's the difference between a tensor's .data and its .grad? What makes a tensor a "leaf"?](#p2-whats-the-difference-between-a-tensors-data-and-its-grad-what-makes-a-tensor-a-leaf)
- [P3: What does .to(device) actually do under the hood, and is the copy synchronous or asynchronous?](#p3-what-does-todevice-actually-do-under-the-hood-and-is-the-copy-synchronous-or-asynchronous)
- [P4: What's the difference between in-place ops (relu_) and out-of-place ones? Why does autograd warn about in-place?](#p4-whats-the-difference-between-in-place-ops-relu_-and-out-of-place-ones-why-does-autograd-warn-about-in-place)
- [P5: What is broadcasting, and what are its memory implications (does it allocate, or is it a view)?](#p5-what-is-broadcasting-and-what-are-its-memory-implications-does-it-allocate-or-is-it-a-view)
- [P6: What's the difference between contiguous and non-contiguous tensors? When does .contiguous() matter?](#p6-whats-the-difference-between-contiguous-and-non-contiguous-tensors-when-does-contiguous-matter)

### nn.Module mechanics
- [P7: What is nn.Module exactly, and why is __call__ different from forward?](#p7-what-is-nnmodule-exactly-and-why-is-__call__-different-from-forward)
- [P8: What's the difference between a Parameter, a Buffer, and a plain tensor attribute?](#p8-whats-the-difference-between-a-parameter-a-buffer-and-a-plain-tensor-attribute)
- [P9: How does state_dict() work, and what does it preserve / what does it skip?](#p9-how-does-state_dict-work-and-what-does-it-preserve-what-does-it-skip)
- [P10: What does setting an nn.Module to training vs evaluation mode actually change at runtime?](#p10-what-does-setting-an-nnmodule-to-training-vs-evaluation-mode-actually-change-at-runtime-batchnorm-dropout-autograd-state)
- [P11: What does model.to(device) do as it traverses the module tree?](#p11-what-does-modeltodevice-do-as-it-traverses-the-module-tree)
- [P12: What's the difference between children(), modules(), named_modules()?](#p12-whats-the-difference-between-children-modules-named_modules)

### Autograd
- [P13: Is PyTorch's computation graph static or dynamic? What does that imply for instrumentation?](#p13-is-pytorchs-computation-graph-static-or-dynamic-what-does-that-imply-for-instrumentation)
- [P14: What does loss.backward() actually do, step by step?](#p14-what-does-lossbackward-actually-do-step-by-step)
- [P15: What's the difference between .grad, .grad_fn, and the autograd engine's task graph?](#p15-whats-the-difference-between-grad-grad_fn-and-the-autograd-engines-task-graph)
- [P16: Why do gradients accumulate by default? When is that useful, when is it a footgun?](#p16-why-do-gradients-accumulate-by-default-when-is-that-useful-when-is-it-a-footgun)
- [P17: When do you need retain_graph=True?](#p17-when-do-you-need-retain_graphtrue)
- [P18: What's the difference between detach(), no_grad(), inference_mode(), and requires_grad=False?](#p18-whats-the-difference-between-detach-no_grad-inference_mode-and-requires_gradfalse)
- [P19: How does a tensor hook (register_hook) differ from a module hook (register_forward_hook)?](#p19-how-does-a-tensor-hook-register_hook-differ-from-a-module-hook-register_forward_hook)

### Optimizers
- [P20: What state does an Adam-class optimizer hold per parameter, and why?](#p20-what-state-does-an-adam-class-optimizer-hold-per-parameter-and-why)
- [P21: What does optimizer.step() actually do, and how does it know which gradients to use?](#p21-what-does-optimizerstep-actually-do-and-how-does-it-know-which-gradients-to-use)
- [P22: What are param_groups for, and when do you reach for them?](#p22-what-are-param_groups-for-and-when-do-you-reach-for-them-different-lrs-per-layer-weight-decay-groups)
- [P23: Why prefer zero_grad(set_to_none=True) over the old default?](#p23-why-prefer-zero_gradset_to_nonetrue-over-the-old-default)

### DataLoader
- [P24: What's the relationship between Dataset, Sampler, and DataLoader?](#p24-whats-the-relationship-between-dataset-sampler-and-dataloader)
- [P25: How does num_workers > 0 actually work — threads, processes, IPC?](#p25-how-does-num_workers-0-actually-work-threads-processes-ipc)
- [P26: What is pin_memory=True and when does it help?](#p26-what-is-pin_memorytrue-and-when-does-it-help)
- [P27: What's collate_fn, and when do you need a custom one?](#p27-whats-collate_fn-and-when-do-you-need-a-custom-one)
- [P28: What is persistent_workers, and what problem does it solve?](#p28-what-is-persistent_workers-and-what-problem-does-it-solve)
- [P29: Why does DataLoader sometimes stall a training step, and what does TraceML look at to diagnose this?](#p29-why-does-dataloader-sometimes-stall-a-training-step-and-what-does-traceml-look-at-to-diagnose-this)

### CUDA dispatch path & memory model
- [P30: How does y = torch.relu(x) get from Python all the way to a CUDA kernel?](#p30-how-does-y-torchrelux-get-from-python-all-the-way-to-a-cuda-kernel-the-dispatcher-path)
- [P31: What is ATen, and what is the C++ dispatcher? Why does it matter for instrumentation?](#p31-what-is-aten-and-what-is-the-c-dispatcher-why-does-it-matter-for-instrumentation)
- [P32: What does torch.cuda.synchronize() actually do, and why is it expensive?](#p32-what-does-torchcudasynchronize-actually-do-and-why-is-it-expensive)
- [P33: How does PyTorch report GPU memory: memory_allocated, max_memory_allocated, the caching allocator?](#p33-how-does-pytorch-report-gpu-memory-memory_allocated-max_memory_allocated-the-caching-allocator)
- [P34: Why is reported memory sometimes lower than nvidia-smi shows? (caching allocator, fragmentation)](#p34-why-is-reported-memory-sometimes-lower-than-nvidia-smi-shows-caching-allocator-fragmentation)

### Mixed precision (AMP)
- [P35: What does torch.autocast actually wrap, and how does it pick which ops to downcast?](#p35-what-does-torchautocast-actually-wrap-and-how-does-it-pick-which-ops-to-downcast)
- [P36: Why does fp16 need a GradScaler, but bf16 doesn't?](#p36-why-does-fp16-need-a-gradscaler-but-bf16-doesnt)
- [P37: When does AMP cause NaN losses or silent precision loss, and how do you debug it?](#p37-when-does-amp-cause-nan-losses-or-silent-precision-loss-and-how-do-you-debug-it)

### Distributed (DDP / FSDP)
- [P38: What does torch.distributed.init_process_group() actually set up?](#p38-what-does-torchdistributedinit_process_group-actually-set-up)
- [P39: What does DistributedDataParallel wrap around your model? How is gradient sync inserted into backward?](#p39-what-does-distributeddataparallel-wrap-around-your-model-how-is-gradient-sync-inserted-into-backward)
- [P40: What is gradient bucketing, and what does bucket_cap_mb tune?](#p40-what-is-gradient-bucketing-and-what-does-bucket_cap_mb-tune)
- [P41: What's the difference between DDP, FSDP, and ZeRO (DeepSpeed)?](#p41-whats-the-difference-between-ddp-fsdp-and-zero-deepspeed)
- [P42: What happens if one rank hangs? How does NCCL detect / recover, and what's the role of NCCL_TIMEOUT?](#p42-what-happens-if-one-rank-hangs-how-does-nccl-detect-recover-and-whats-the-role-of-nccl_timeout)

### Eager vs graph mode (torch.compile, JIT, fx)
- [P43: What does torch.compile actually do? How is it different from TorchScript?](#p43-what-does-torchcompile-actually-do-how-is-it-different-from-torchscript-the-older-torchjitscript)
- [P44: What are TorchDynamo, AOT autograd, and torch.fx — how do they relate?](#p44-what-are-torchdynamo-aot-autograd-and-torchfx-how-do-they-relate)
- [P45: Does torch.compile break module hooks / TraceML's instrumentation? If so, what's the workaround?](#p45-does-torchcompile-break-module-hooks-tracemls-instrumentation-if-so-whats-the-workaround)

### Checkpointing & state
- [P46: What is actually inside a saved checkpoint file (format, contents)?](#p46-what-is-actually-inside-a-saved-checkpoint-file-format-contents)
- [P47: What does torch.save / torch.load use under the hood, and what are the security implications?](#p47-what-does-torchsave-torchload-use-under-the-hood-and-what-are-the-security-implications-around-loading-untrusted-checkpoints)

### PyTorch internals relevant to TraceML
- [P48: What is _call_impl, and why does TraceML monkey-patch around it?](#p48-what-is-_call_impl-and-why-does-traceml-monkey-patch-around-it-instead-of-just-using-public-hooks)
- [P49: What's the exact firing order of forward_pre_hook, forward_hook, backward_pre_hook, backward_hook?](#p49-whats-the-exact-firing-order-of-forward_pre_hook-forward_hook-backward_pre_hook-backward_hook)
- [P50: How does PyTorch's built-in profiler differ from TraceML's approach?](#p50-how-does-pytorchs-built-in-profiler-torchprofiler-differ-from-tracemls-approach-where-does-traceml-do-better-and-where-does-the-profiler-do-better)
- [P51: Which torch.cuda.* APIs does TraceML rely on, and how stable are they across PyTorch versions?](#p51-which-torchcuda-apis-does-traceml-rely-on-and-how-stable-are-they-across-pytorch-versions)
- [P52: How does TraceML measure per-layer memory, and what's the relationship to torch.cuda.memory_allocated?](#p52-how-does-traceml-measure-per-layer-memory-and-whats-the-relationship-to-torchcudamemory_allocated)

---

## Tensor fundamentals

### P1: What is a `torch.Tensor` actually made of (storage, strides, dtype, device, view vs copy)?

**Date:** 2026-04-24

**Short answer:** A `torch.Tensor` is a thin **view object** over a 1D contiguous chunk of memory called a **`Storage`**. The view consists of a **dtype** (element type), a **device** (where the storage lives — CPU RAM, `cuda:0`, etc.), a **shape** (logical dimensions), a **stride tuple** (how many elements to skip per dimension), and an **offset** into storage. Operations like `.transpose`, `.view`, `.reshape`, `.permute`, slicing, and indexing usually produce a **new Tensor object that shares the same Storage** — that is a **view**. Operations like `.clone()`, `.to()` to a different device, or any op that needs differently-laid-out output produce a **copy** with its own fresh storage.

**Long answer:**

*Two-layer model: Tensor on top of Storage.* PyTorch's tensor is split into two C++ objects:

- `at::Storage` — a flat, typeless byte buffer with a size, an allocator, and a device. Roughly `void* data`, `size_t nbytes`, `Device device`, `Allocator* alloc`. No knowledge of shape or dtype.
- `at::Tensor` (more precisely `TensorImpl`) — a header that points at a Storage and adds dtype, shape, strides, offset, plus autograd metadata.

You can have many Tensors pointing into the same Storage at different offsets and strides. This is exactly how zero-copy slicing works: `x[1:3]` creates a new TensorImpl with a different offset and the same Storage pointer. Memory is shared; mutations through one view are visible through the other.

```python
import torch
x = torch.arange(12).reshape(3, 4)        # one storage, 12 ints
y = x[1:, :2]                             # view: shares storage
y[0, 0] = 999
print(x[1, 0])                            # -> 999
print(x.storage().data_ptr() == y.storage().data_ptr())  # True
```

*Strides — the key to views.* Strides are how PyTorch turns an N-dimensional logical index into a 1D storage offset. For a 2D tensor of shape `(R, C)` with strides `(s_r, s_c)`, the element at logical index `(i, j)` lives at storage offset `offset + i*s_r + j*s_c` (in element-units, not bytes). For a freshly allocated `(3, 4)` int64 tensor, the strides are `(4, 1)` — moving down a row means jumping 4 elements, moving across a column means jumping 1.

`.transpose(0, 1)` doesn't move any data — it just swaps the entries in the strides tuple to `(1, 4)`. The storage is unchanged; the new tensor walks the same memory in a different order. This is what makes transpose O(1).

```python
x = torch.arange(12).reshape(3, 4)        # shape (3,4), strides (4,1)
xt = x.t()                                # shape (4,3), strides (1,4)
xt.is_contiguous()                        # False
xt.storage().data_ptr() == x.storage().data_ptr()  # True
```

This is also why a transposed tensor is **non-contiguous** — see [P6](#p6-whats-the-difference-between-contiguous-and-non-contiguous-tensors-when-does-contiguous-matter) for the consequences.

*Dtype.* The element type — `torch.float32`, `torch.float16`, `torch.bfloat16`, `torch.int64`, `torch.bool`, etc. The Tensor header stores the dtype; the Storage is dtype-agnostic raw bytes. `.to(dtype=torch.float16)` allocates a new Storage half the size and copies converted values.

*Device.* Each Storage lives on exactly one device. CPU storages are allocated by the libc allocator (or the pinned allocator for `pin_memory=True`); CUDA storages are allocated by the **caching allocator** in `c10/cuda/CUDACachingAllocator.cpp`. A Tensor on `cuda:0` cannot share storage with one on `cpu` or `cuda:1` — moving requires copy. See [P3](#p3-what-does-todevice-actually-do-under-the-hood-and-is-the-copy-synchronous-or-asynchronous).

*View vs copy — the operational rule.* The mental shortcut:

- **View:** indexing with slices/ints, `.view()`, `.transpose()`, `.permute()`, `.unsqueeze()`, `.squeeze()`, `.expand()`, `.t()`, `.diagonal()`, `.narrow()`, `.detach()`. All O(1), share storage.
- **Copy:** `.clone()`, `.contiguous()` *if not already contiguous*, `.to()` *if it changes device or dtype*, `.reshape()` *if it can't be done as a view*, fancy indexing (`x[[0, 2, 5]]` or boolean mask), arithmetic (`x + 1` allocates a new tensor).

`.reshape()` is the diplomatic case: it returns a view if possible, else falls back to a copy. `.view()` is strict — raises if a view isn't possible.

*Why this two-layer design exists.* It mirrors NumPy's `ndarray` over `buffer`. The benefit is that all the cheap tensor manipulations (slice, transpose, broadcast, unsqueeze) cost zero memory and zero memcpy — they're metadata edits. Real cost is paid only when something actually needs the data laid out differently (a kernel that requires contiguous input) or moved (different device).

*Autograd add-ons in the Tensor header.* On top of storage/shape/strides/dtype/device, a Tensor also carries:

- `requires_grad: bool`
- `grad: Optional[Tensor]` — accumulated gradient, allocated on first backward
- `grad_fn: Optional[Function]` — pointer to the autograd graph node that produced this tensor (None for leaves; see [P2](#p2-whats-the-difference-between-a-tensors-data-and-its-grad-what-makes-a-tensor-a-leaf))
- `version_counter: int` — incremented on every in-place mutation, used by autograd to detect "you mutated something I needed for backward" (see [P4](#p4-whats-the-difference-between-in-place-ops-relu_-and-out-of-place-ones-why-does-autograd-warn-about-in-place))

So a Tensor is really *(view metadata) + (autograd metadata) + (pointer to Storage)*. Three concerns layered on the same object.

*ML/RL analogy.* If you've used NumPy's structured arrays or written a replay buffer that holds `(s, a, r, s')` as parallel arrays with a circular index, you're already thinking in terms of "one underlying buffer, many logical views." The replay buffer's `.sample(idx)` doesn't allocate — it returns a slice. A PyTorch view is the same idea hardened into the type system.

**Concepts introduced:** `at::Storage` vs `at::Tensor`/`TensorImpl`, dtype, device, shape, strides, offset, view vs copy semantics, contiguous-by-default layout, zero-copy transpose, caching allocator (CUDA), `requires_grad`, `grad_fn`, version counter.

---

### P2: What's the difference between a tensor's `.data` and its `.grad`? What makes a tensor a "leaf"?

**Date:** 2026-04-24

**Short answer:** `.data` is the **values** of the tensor — the underlying storage exposed as a plain tensor *with autograd stripped off*. `.grad` is a **separate tensor of the same shape** that accumulates partial derivatives during `backward()`. A **leaf tensor** is one that was *not* produced by a differentiable operation on other tensors that require grad — i.e., it sits at the bottom of the autograd graph. In practice, leaves are: model parameters, tensors you create directly (`torch.randn(...)`, dataset batches), and any tensor created with `requires_grad=False`. Only leaf tensors with `requires_grad=True` get a `.grad` populated by `backward()`.

**Long answer:**

*The autograd graph view.* When you do `y = x * 2; z = y + 1`, autograd builds a small DAG behind the scenes:

```
x (leaf, requires_grad=True)
   |
   v  MulBackward
y (non-leaf, grad_fn=MulBackward)
   |
   v  AddBackward
z (non-leaf, grad_fn=AddBackward)
```

Each non-leaf tensor has a `grad_fn` pointing to the autograd Function node that made it. Leaves have `grad_fn = None` and (if they require grad) a `grad` slot ready to receive gradients.

`z.backward()` walks this DAG in reverse, calling each `grad_fn`'s backward, and accumulates the result into the `.grad` of every leaf with `requires_grad=True`. Non-leaves don't get `.grad` populated (you'd just get a warning and `None`); if you want intermediate gradients, you call `.retain_grad()` on them before backward.

*Why "leaf".* Imagine the autograd graph as a tree (it's actually a DAG, but the analogy holds). The output (loss scalar) is the root; the inputs (parameters, data) are the leaves. Backprop flows from root to leaves. Leaves are where the derivative chain terminates — there's nothing further upstream to differentiate through, so you can store the result there.

A tensor is a leaf if any of:

- It was created directly: `torch.zeros(3, requires_grad=True)`, `nn.Parameter(...)`, output of `torch.from_numpy(...)`.
- It came in via `torch.no_grad()` or has `requires_grad=False` (in which case it's a leaf but won't have `.grad`).
- It's the result of an op where *no input* required grad.

Quick check: `t.is_leaf` returns True/False.

```python
x = torch.randn(3, requires_grad=True)
w = torch.randn(3, requires_grad=True)
y = x * w                  # not a leaf — has grad_fn=MulBackward
loss = y.sum()
loss.backward()

x.is_leaf, w.is_leaf, y.is_leaf       # True, True, False
x.grad is not None, w.grad is not None # True, True
y.grad                                  # None + warning if you check it
```

*What `.data` actually is.* `t.data` returns a *new tensor that shares the same storage as `t`* but with `requires_grad=False` and `grad_fn=None`. It's essentially "give me the values without the autograd hookup." Historically (pre-0.4) it was the only way to do non-differentiable surgery on a tensor.

The modern recommended replacement is `.detach()`:

- `.detach()` — same shared storage, autograd-stripped, but with proper version-counter tracking. If you mutate the detached tensor in place, autograd notices and errors out at backward time instead of silently giving you a wrong gradient.
- `.data` — same shared storage, autograd-stripped, *bypasses version tracking*. If you mutate `t.data`, autograd doesn't know, and you can silently corrupt gradients.

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
x.data.zero_()       # silently overwrites — autograd will compute wrong grads
x.detach().zero_()   # also overwrites, but increments x's version counter
                     # if x was used in an autograd op, backward() will raise
```

So `.data` is essentially deprecated for user code. Treat it as "the legacy way." Use `.detach()` for "values without grad" and `with torch.no_grad():` for "do this whole block without recording autograd."

*What `.grad` actually is.* It's an *attribute*, not a method — `param.grad` is either `None` (no backward has accumulated into it yet) or a `Tensor` of the same shape, dtype, and device as `param`. It's allocated lazily on the first call to `backward()` that touches this leaf. Subsequent backward calls **add** to it (this is why training loops do `optimizer.zero_grad()` or `param.grad.zero_()` between steps — to prevent silent accumulation across batches).

Important: `param.grad` is itself a regular tensor sitting in memory. For a 7B-parameter model in fp32, the grads alone are ~28 GB — same as the parameters. This is why mixed-precision and gradient checkpointing matter for large models.

*The `nn.Parameter` distinction.* `nn.Parameter(tensor)` is just a `Tensor` subclass with two behaviors: `requires_grad=True` by default, and when assigned as an attribute on an `nn.Module`, it's auto-registered in `module.parameters()`. That's it — no other magic. Parameters are leaves.

*RL analogy.* Think of the leaves as the things you'll update with the optimizer (the actor's weights, the critic's weights). The non-leaves are intermediate activations in the rollout — you don't update them, they only exist so backprop can route gradients through them. `.grad` on a parameter is the policy gradient or value gradient ready for the optimizer step.

*Connection to in-place ops.* The version counter mentioned in [P1](#p1-what-is-a-torchtensor-actually-made-of-storage-strides-dtype-device-view-vs-copy) is what makes `.detach()` safe and `.data` unsafe. `.data` is the back door that skips the safety check. See [P4](#p4-whats-the-difference-between-in-place-ops-relu_-and-out-of-place-ones-why-does-autograd-warn-about-in-place) for the in-place story.

**Concepts introduced:** autograd graph as DAG, leaf vs non-leaf tensor, `grad_fn`, `.grad` attribute, gradient accumulation across backward calls, `.data` (legacy, unsafe) vs `.detach()` (safe), version counter, `torch.no_grad()`, `nn.Parameter` registration, `retain_grad()` for intermediates.

---

### P3: What does `.to(device)` actually do under the hood, and is the copy synchronous or asynchronous?

**Date:** 2026-04-24

**Short answer:** `tensor.to(device)` allocates a fresh storage on the target device and **memcpys** the bytes across. CPU→CPU is a host memcpy; CPU→GPU and GPU→CPU go through `cudaMemcpy` / `cudaMemcpyAsync`; GPU→GPU goes through `cudaMemcpyPeer` (or NVLink/PCIe). By default, GPU-touching transfers are queued on the current CUDA stream and are **asynchronous from Python's perspective** — Python returns immediately, the actual copy happens later on the GPU. To make this overlap useful, the source CPU memory must be **pinned** (`pin_memory=True`) and you must pass `non_blocking=True`; otherwise PyTorch silently synchronizes to keep things correct.

**Long answer:**

*The general flow.* `t.to(device)` does roughly:

1. If `t` is already on `device` and no dtype change, return `t` itself (no-op).
2. Allocate a new Storage on the target device with the right number of bytes.
3. Issue the appropriate memcpy primitive between source and destination buffers.
4. Construct a new Tensor view (same shape, strides, dtype) over the new storage.
5. Return the new Tensor. Autograd records a `CopyBackward` if `requires_grad=True`.

The interesting part is step 3, which differs by source/destination pair.

*CPU → CPU.* Plain libc-style memcpy on the host. Synchronous and unsurprising.

*CPU → GPU (host-to-device, H2D).* Goes through CUDA's `cudaMemcpyAsync` from the source host buffer to the destination device buffer, queued on the current stream. But there's a catch: **for the copy to be truly asynchronous, the source host buffer must be pinned (page-locked) memory.** Otherwise:

- The OS can page out unpinned memory at any moment. The GPU's DMA engine cannot tolerate that — it needs the bytes at a fixed physical address for the duration of the transfer.
- So if the source is pageable (normal `malloc`'d) memory, the CUDA driver internally allocates a pinned staging buffer, memcpys host→staging synchronously on the CPU, *then* DMAs staging→device. The host-side CPU work blocks until the staging memcpy is done — your Python thread waits.
- If the source is already pinned (`tensor.pin_memory()` or `DataLoader(pin_memory=True)`), the DMA engine reads directly from your buffer, and the Python call returns immediately while the GPU does the transfer in the background.

This is why `tensor.cuda(non_blocking=True)` only actually overlaps with compute when paired with pinned memory. Without pinning, `non_blocking=True` is silently ignored. See [Q15](learning-qa.md#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread) for the stream-level mechanics — the H2D copy gets queued on a stream, and you can use a separate stream from compute to overlap data prep with the previous step's training.

```python
# pinned + non_blocking → real overlap
batch = batch.pin_memory()
x = batch.to('cuda', non_blocking=True)   # returns immediately
out = model(prev_x)                        # runs in parallel with the H2D
```

*GPU → CPU (device-to-host, D2H).* `cudaMemcpyAsync` in the other direction. By default, this **synchronizes** the calling thread — even with `non_blocking=True`, PyTorch will sync the stream before the call returns *unless* the destination is pinned. Reason: if Python gets back a "tensor on CPU" but the bytes aren't actually written yet, any subsequent CPU code reading those values would get garbage. PyTorch errs on the side of correctness.

This is why "GPU→CPU" copies show up in profilers as a frequent cause of host-side stalls. `loss.item()`, `tensor.cpu()`, `tensor.numpy()` all force a sync on the underlying CUDA stream.

*GPU → GPU, same node.* `cudaMemcpyPeer` if the GPUs are peer-accessible (typically true within an NVLink/NVSwitch domain or even over PCIe). Async on the source stream. This is what `tensor.to('cuda:1')` does. NCCL `all-reduce` and DDP gradient sync use a more sophisticated path on top of this primitive — see [Q12](learning-qa.md#q12-what-is-nccl-all-reduce-and-why-is-it-a-barrier).

*GPU → GPU, different nodes.* Goes through host RAM and the network (Ethernet, Infiniband — see [Q14](learning-qa.md#q14-what-is-rdma-infiniband-and-why-does-it-matter-for-multi-node-training)). PyTorch's `.to()` doesn't directly do this — you'd use NCCL or `torch.distributed.send/recv`.

*The "is it async?" cheat sheet.*

| Direction | Sync default? | Truly async needs |
| --------- | ------------- | ----------------- |
| CPU → CPU | Sync          | n/a (always sync) |
| CPU → GPU | Sync          | Pinned source + `non_blocking=True` |
| GPU → CPU | Sync          | Pinned dest + `non_blocking=True` (rare) |
| GPU → GPU (same node) | Async on source stream | Already async; correctness via stream ordering |

*The dtype-change case.* `.to(dtype=torch.float16)` allocates new storage even on the same device, because float32 and float16 have different element sizes — there's no in-place conversion. Same logic for `.to(device='cuda', dtype=torch.float16)`: one new allocation on the GPU, dtype conversion happens as part of the kernel that does the memcpy.

*What "current stream" means here.* By default, transfers go on whatever the current CUDA stream is for this CPU thread (usually the default stream). If you want H2D to overlap with compute, you'd put the transfer on a *different* stream:

```python
copy_stream = torch.cuda.Stream()
with torch.cuda.stream(copy_stream):
    batch_gpu = batch.to('cuda', non_blocking=True)
# main stream keeps doing compute on the previous batch
torch.cuda.current_stream().wait_stream(copy_stream)  # before using batch_gpu
```

This is the "double-buffered dataloader" pattern. Most users don't write this by hand — `DataLoader(pin_memory=True)` plus `non_blocking=True` gives you something close enough.

*Implications for TraceML's bottleneck attribution.* If TraceML's step-time decomposition shows a long "dataloader" or "H2D" segment, the user is probably:

- Not pinning memory in their DataLoader → every batch transfer goes through the synchronous staging path.
- Calling `.cuda()` without `non_blocking=True` → no overlap even if pinned.
- Hitting a `loss.item()` or `.cpu()` mid-step → forces a D2H sync that drains the queue.

These are some of the most common "cheap fix, big win" diagnostics — exactly what TraceML aims to surface automatically.

*RL analogy.* Think of `non_blocking=True` + pinned memory like firing off an async env step in a vectorized environment: the actor can compute the next action while the env is stepping. Without pinning, you're effectively stuck in the synchronous version where the actor blocks on each `env.step()`.

**Concepts introduced:** `cudaMemcpy` vs `cudaMemcpyAsync`, pageable vs pinned (page-locked) host memory, DMA engine, `pin_memory=True`, `non_blocking=True`, host-side stalls from `loss.item`/`.cpu()`/`.numpy`, `cudaMemcpyPeer` for GPU↔GPU, dtype-change forces reallocation, double-buffered dataloader pattern, CopyBackward autograd node, current-stream semantics for transfers.

---

### P4: What's the difference between in-place ops (`relu_`) and out-of-place ones? Why does autograd warn about in-place?

**Date:** 2026-04-24

**Short answer:** An **out-of-place** op (`y = x.relu()`) allocates a new tensor for the result and leaves `x` untouched. An **in-place** op (`x.relu_()`, the trailing underscore is the convention) overwrites `x`'s storage with the result. In-place saves memory but is dangerous for autograd: if `x` was needed to compute a gradient *later* during backward, overwriting it corrupts the computation. PyTorch's **version counter** detects this — every in-place op bumps the tensor's version, and at backward time autograd checks "is this tensor's version the same as when I saved it?" If not, it raises `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.

**Long answer:**

*The naming convention.* PyTorch uses two parallel APIs:

- Functional / out-of-place: `torch.relu(x)`, `x.relu()`, `x + y`, `x.add(y)`, `x.mul(y)`. Returns a new tensor.
- In-place: `x.relu_()`, `x.add_(y)`, `x += y`, `x.mul_(y)`, `x.zero_()`, `x.copy_(y)`. Mutates `x`, returns `x` (so it can be chained but you usually ignore the return).

The trailing underscore is the universal flag: any method ending in `_` mutates the receiver. Operators have their own pairs: `+` is out-of-place, `+=` is in-place.

*Why in-place exists at all.* Memory. A forward pass through a deep network creates a lot of intermediate activations. If every elementwise op (relu, dropout, layernorm scale) allocated a new tensor, peak memory would balloon. In-place ops let you reuse the input buffer, which can meaningfully shrink peak activation memory.

Concrete example — a ResNet block:

```python
out = self.conv1(x)
out = self.bn1(out)
out = F.relu(out, inplace=True)   # reuses bn1's output buffer
out = self.conv2(out)
out = self.bn2(out)
out += identity                    # in-place add of skip connection
out = F.relu(out, inplace=True)
```

Two in-place relus and one in-place add. Each saves an activation-sized allocation, multiplied across all blocks and batch dims. For training large models on tight VRAM, this matters.

*Why autograd has a problem with it.* Autograd needs to save certain inputs/outputs from the forward pass to compute gradients during backward. The function `f(x) = x²` needs `x` saved so backward can compute `2*x`. Sigmoid needs the *output* `y = sigmoid(x)` saved so backward can compute `y * (1 - y)`. Different ops save different things.

If you compute `y = sigmoid(x)` and then later do `y.zero_()` before backward, the saved value is corrupted — autograd would compute a wrong gradient, silently. To catch this, PyTorch maintains a **version counter** on each Tensor's storage:

- Every in-place op on a tensor increments the version counter.
- When autograd saves a tensor for backward, it also records the version-at-save-time.
- At backward time, autograd checks: does the saved version match the current version? If not → raise.

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2          # autograd saves x, version = 0
x.add_(1)           # version bumped to 1
y.sum().backward()  # RuntimeError: one of the variables needed for
                    # gradient computation has been modified by an
                    # inplace operation [version 1 vs expected 0]
```

This is fail-loud, not fail-silent. Without the version counter you'd get NaNs or silently wrong gradients three weeks into a training run.

*When in-place is safe.* In-place is safe when *no autograd Function has saved this tensor.* Common safe cases:

- Mutating a tensor that doesn't `requires_grad` — no autograd graph involves it.
- Mutating a tensor inside `torch.no_grad()` — no graph being built.
- Mutating a tensor whose only future use is being fed into ops that don't need to save it.
- The activation reuse pattern above: `relu(inplace=True)` is safe because relu's backward needs only the *sign* of the input, which it can derive from the output (relu's output is zero exactly where the input was negative). PyTorch's relu backward saves the output, not the input, specifically to enable inplace.

*Optimizer-side in-place.* `optimizer.step()` is one giant in-place update — `param.data -= lr * param.grad` (and similar in Adam/AdamW with momentum buffers). This is fine because:

- It runs *after* `backward()`, so all the saved tensors have already been consumed.
- It's done outside any autograd-tracked region (`torch.no_grad()` internally, or via the parameter's `.data` / `with torch.no_grad():` guard the optimizer uses).
- Parameters are leaves; mutating a leaf doesn't break the autograd DAG of *future* forward passes (the next forward builds a new graph from scratch).

This is also why optimizer state (momentum, variance) lives on `.data` / outside autograd — it's all in-place math we don't want recorded.

*Connection to `.data` vs `.detach()`.* From [P2](#p2-whats-the-difference-between-a-tensors-data-and-its-grad-what-makes-a-tensor-a-leaf): `.data` bypasses the version counter, `.detach()` respects it. So `x.data.zero_()` silently corrupts; `x.detach().zero_()` increments x's version and triggers the safety net at backward time.

*The autograd graph perspective.* Recall from [P2](#p2-whats-the-difference-between-a-tensors-data-and-its-grad-what-makes-a-tensor-a-leaf) that the autograd graph is a DAG with `grad_fn` nodes. Each `grad_fn` may hold references to tensors saved during forward (`saved_tensors`). The version counter check is a runtime invariant: "the bytes I'm about to consume are still the bytes I was promised." This is the same kind of invariant a STM (software transactional memory) system enforces — read-set validation at commit time.

*RL analogy.* Picture a replay buffer where you save `(s, a, r, s')` tuples. If a background thread later mutates the contents of one of the saved transitions, your gradient update on that batch would be junk. The version counter is PyTorch's way of saying "you can mutate, but I'll catch you if you mutate something I need." It's more graceful than locking, more reliable than convention.

*Practical guidance.*

- Default to out-of-place. It's almost always fast enough and never wrong.
- Reach for in-place when peak memory is the bottleneck and you can prove the tensor isn't needed for backward.
- Library-blessed patterns (`F.relu(inplace=True)`, optimizer step) are safe — they're audited.
- If you see "modified by an inplace operation" in training, the fix is almost always: change one `x += y` to `x = x + y` and re-run. The error message tells you which version mismatch fired; trace backward from there.

**Concepts introduced:** in-place op naming convention (trailing `_`), out-of-place vs in-place semantics, peak activation memory, version counter on Storage, `saved_tensors` in autograd Functions, "needed for gradient computation" runtime check, why `relu(inplace=True)` specifically is safe (output-based backward), optimizer step as bulk in-place under `no_grad`, STM-style read-set validation analogy.

---

### P5: What is broadcasting, and what are its memory implications (does it allocate, or is it a view)?

**Date:** 2026-04-24

**Short answer:** **Broadcasting** is the rule that lets you do arithmetic between tensors of different but *compatible* shapes by virtually expanding the smaller one to match the larger. Compatibility: align shapes from the right; each pair of dims must be equal, or one of them must be 1, or one must be missing. The expansion itself is **a free, zero-copy view** — `tensor.expand()` creates a new TensorImpl with **stride 0** along the broadcast dims, so reading element `[i, j]` from a "size 1 expanded to size N" dim always reads the same underlying byte. The result tensor produced by the arithmetic op (e.g., `x + y`) *is* a fresh allocation of the broadcast-shaped output. So broadcasting itself is free; the operation that consumes it usually isn't.

**Long answer:**

*The rule, precisely.* Given two tensor shapes, align them from the rightmost dimension and walk left:

- If both dims are equal → keep that size.
- If one is 1 → broadcast to the other.
- If one is missing → treat as 1, broadcast to the other.
- Else → error.

```python
torch.zeros(3, 4) + torch.zeros(4)         # (3,4) + (4,) -> (3,4)
torch.zeros(3, 4) + torch.zeros(3, 1)      # (3,4) + (3,1) -> (3,4)
torch.zeros(3, 4) + torch.zeros(1, 4)      # (3,4) + (1,4) -> (3,4)
torch.zeros(3, 4) + torch.zeros(2, 4)      # error: 3 vs 2
```

Common ML uses: adding a per-channel bias `(C,)` to an activation `(N, C, H, W)`. Subtracting a per-batch mean `(N, 1)` from `(N, D)`. Outer products `(N, 1) * (1, M) → (N, M)`.

*The zero-stride trick.* Internally, broadcasting is implemented via `torch.expand`, which produces a view with **stride 0** along the expanded dimensions. Recall from [P1](#p1-what-is-a-torchtensor-actually-made-of-storage-strides-dtype-device-view-vs-copy) that an N-D index `(i, j)` translates to a storage offset via `i*stride_0 + j*stride_1`. If `stride_0 = 0`, then the row index `i` doesn't move you in storage — every row reads the same single value.

```python
x = torch.tensor([1., 2., 3.])             # shape (3,), strides (1,)
xb = x.expand(4, 3)                         # shape (4, 3), strides (0, 1)
xb.storage().data_ptr() == x.storage().data_ptr()  # True
xb.is_contiguous()                          # False
```

`xb` looks like a 4×3 matrix where every row is `[1, 2, 3]`. The storage is still just 3 floats. Element `xb[i, j]` resolves to `storage[0 + i*0 + j*1] = storage[j]`. Free memory, free copy.

This is the same trick NumPy and JAX use. It's not a PyTorch invention — it dates back to NumPy's broadcasting design and ultimately to APL.

*Where the allocation happens.* Broadcasting itself is free, but `x + y` still has to produce an output tensor with the broadcast shape. That output is a fresh allocation. So:

- `x.expand(...)` → 0 bytes, returns a view.
- `(x + y)` where shapes differ → allocates output of broadcast shape, runs the elementwise kernel that reads through the zero-stride views.

The kernel itself is broadcasting-aware: it doesn't materialize the expanded view in memory, it just reads through the strided indexing. So a `(3, 4)` + `(4,)` add reads 12 + 4 = 16 floats and writes 12 floats — no intermediate `(3, 4)` copy of the smaller tensor.

*The footgun: in-place broadcast.* You cannot broadcast into the *destination* of an in-place op. If `x` is `(3, 4)` and `y` is `(4,)`, `x += y` works (broadcast on the RHS, output shape matches LHS). But `y += x` fails — broadcasting `y` to `(3, 4)` and writing back into a `(4,)` storage doesn't make sense.

Worse footgun: silent shape mismatches. `loss = (pred - target)**2` where `pred` is `(B, 1)` and `target` is `(B,)` → broadcasts to `(B, B)`, and you're computing the wrong loss. This shows up as a model that "trains" but to garbage. Prefer explicit `.squeeze()` / `.unsqueeze()` to make shapes match.

*Memory implications you actually care about.*

- Broadcasting itself: **free**.
- Output tensor: **allocates broadcast shape**. A `(N, 1, D)` * `(1, M, D)` outer product → `(N, M, D)` allocation. If N=M=10000 and D=128, that's 5 GB at fp32. Easy to OOM.
- `tensor.expand(big_shape)` followed by `.contiguous()`: **materializes the view into a real allocation of the expanded size.** This is the difference between "I have a free view" and "I have an actual copy that my next kernel needs." Some ops require contiguous input — see [P6](#p6-whats-the-difference-between-contiguous-and-non-contiguous-tensors-when-does-contiguous-matter).

*`expand` vs `repeat`.*

- `tensor.expand(...)` — view, zero-stride, no copy. Read-only-ish (you can't safely mutate a stride-0 view).
- `tensor.repeat(...)` — copy, materializes the repeated values into fresh storage. Memory cost = output size.

If you only need to *read* the expanded tensor (e.g., feed it into an arithmetic op), use `expand`. Use `repeat` only when you need an independent, contiguous, mutable copy.

*Stride-0 views and autograd.* You can't safely do in-place ops on a stride-0 view because writing to `xb[0, 0] = 5` and `xb[1, 0] = 7` both target the same storage element — last write wins, and autograd has no clean way to track "I broadcast then mutated." PyTorch raises an error (or warns) if you try.

*ML/RL analogy.* Broadcasting is the same trick as defining a bias term `b` once and conceptually adding it to every row of a feature matrix. Mathematically it's `XW + 1·b^T` where `1` is a column of ones; computationally, you don't materialize the column of ones — you broadcast. PyTorch generalizes this to arbitrary dim alignment.

**Concepts introduced:** broadcasting compatibility rule (right-aligned dims; equal, 1, or missing), `tensor.expand` as zero-stride view, stride-0 trick for free virtual replication, broadcast-aware elementwise kernels (no materialization), allocation cost of the *output* (not the broadcast itself), `expand` vs `repeat` (view vs copy), silent shape-mismatch footgun, in-place broadcast restrictions, autograd issues with mutating stride-0 views.

---

### P6: What's the difference between contiguous and non-contiguous tensors? When does `.contiguous()` matter?

**Date:** 2026-04-24

**Short answer:** A tensor is **contiguous** when its elements are laid out in memory in **row-major (C-style) order with no gaps** — equivalently, when its strides match what you'd compute from its shape (rightmost stride = 1, each leftward stride = previous stride × previous size). Operations that *don't* preserve this — `.transpose()`, `.permute()`, `.expand()`, certain slices — produce non-contiguous views over the same storage. **`.contiguous()`** allocates fresh storage and copies the elements into row-major order, returning a contiguous tensor. You need it when an op requires contiguous input (e.g., `.view()`, many CUDA kernels, `flatten` for some paths, sending to NCCL, exporting via `numpy()`); otherwise you'll get an error or a hidden copy.

**Long answer:**

*Definition, precisely.* For a tensor with shape `(d_0, d_1, ..., d_{n-1})`, the **contiguous strides** are computed right-to-left:

```
stride[n-1] = 1
stride[k]   = stride[k+1] * d_{k+1}
```

For shape `(3, 4, 5)`, contiguous strides are `(20, 5, 1)`. A tensor with these strides has its elements packed in row-major order: walking the storage linearly visits `[0, 0, 0]`, `[0, 0, 1]`, ..., `[0, 0, 4]`, `[0, 1, 0]`, ..., etc.

`tensor.is_contiguous()` returns True iff the strides match this formula and the storage offset is 0 (so no "starts in the middle" weirdness). `tensor.stride()` gives you the actual strides; compare to `tensor.shape` to see why it's non-contiguous.

*How tensors become non-contiguous.*

- `t.transpose(0, 1)` swaps two strides → non-contiguous if the swapped dims have size > 1.
- `t.permute(2, 0, 1)` reorders strides → non-contiguous in general.
- `t.expand(...)` introduces stride-0 dims → non-contiguous (see [P5](#p5-what-is-broadcasting-and-what-are-its-memory-implications-does-it-allocate-or-is-it-a-view)).
- `t[:, ::2]` (strided slice) → non-contiguous because stride along the sliced dim is 2× larger than the natural one.
- `t.unsqueeze(0)` adds a new dim of size 1 → still contiguous (size-1 dims don't break contiguity).
- `t.view(new_shape)` → contiguous if input was contiguous; **fails outright** if input is non-contiguous and the new shape can't be expressed without a copy.

```python
x = torch.arange(12).reshape(3, 4)         # contiguous, strides (4, 1)
xt = x.t()                                  # shape (4, 3), strides (1, 4)
x.is_contiguous(), xt.is_contiguous()       # True, False
xt.view(12)                                 # RuntimeError: not compatible
xt.contiguous().view(12)                    # OK, copy then view
xt.reshape(12)                              # OK — reshape silently copies
```

*Why kernels care.* Most CUDA kernels (and many CPU SIMD kernels) are written assuming row-major contiguous input because:

- They use linear indexing (`storage[i]`) to vectorize memory loads, which only matches your tensor's logical layout if it's contiguous.
- GPU memory access is fastest when consecutive threads in a warp read consecutive bytes ("coalesced access"). Stride-1 along the inner dim ≈ coalesced; large or zero strides break this.
- Arbitrary strides force the kernel to use a slower indexing path or pay an indexing-math overhead per element.

So PyTorch will:

- Either silently call `.contiguous()` for you (allocates and copies — invisible perf hit).
- Or refuse and tell you `.contiguous()` is required (`view`, `flatten` in some cases, `.numpy()`, NCCL ops).
- Or accept non-contiguous input but route through a generic, slower kernel.

The middle case is the dangerous one for performance: a transposed activation flowing into a matmul might copy 2 GB silently every step.

*`view` vs `reshape` revisited.*

- `view` is a strict alias: returns a view if and only if the new shape can be expressed by re-striding existing storage. Otherwise raises.
- `reshape` is `view` with a fallback: try to view; if not possible, return `contiguous().view(...)`. Costs a copy in the fallback case.

So `reshape` is more permissive but can hide allocations. `view` is loud but predictable. Power users prefer `view` precisely because the failure mode is visible.

*When `.contiguous()` is necessary.*

- Right before `view(...)` on a previously transposed/permuted tensor.
- Before `tensor.numpy()` if the tensor is non-contiguous (NumPy can sometimes handle strided arrays but PyTorch's `.numpy()` requires contiguous).
- Before NCCL collective calls — NCCL requires contiguous send/recv buffers.
- Before some custom CUDA kernels you wrote that assume row-major.
- Before serialization to certain formats (ONNX, some checkpoint paths).

When in doubt: if your tensor flows from a `transpose`/`permute`/`expand` into something that crashes or seems mysteriously slow, try `.contiguous()` and see if it changes the symptom.

*Memory cost.* `.contiguous()` on an already-contiguous tensor is free (returns `self`). On a non-contiguous tensor, it allocates a fresh storage of size `numel * dtype_size` and copies, walking the input via its strides and writing the output in linear order. Cost is one full memcpy of the tensor.

*The `channels_last` plot twist.* PyTorch supports memory-format hints — notably `torch.channels_last` for 4D tensors. A `(N, C, H, W)` tensor in channels-last has strides `(C*H*W, 1, C*W, C)` instead of the default `(C*H*W, H*W, W, 1)`. It's *contiguous in a different sense* — `.is_contiguous(memory_format=torch.channels_last)` returns True even though `.is_contiguous()` returns False.

This matters because some convolution kernels (especially on Tensor Cores, fp16/bf16) are dramatically faster when input is channels-last — the hardware wants the channel dim to be the fast-varying one. So "non-contiguous in the default sense" doesn't always mean "slow"; sometimes it's the *right* layout for the hardware. ResNet/EfficientNet/ConvNeXt training in mixed precision often gets a 1.5–2× throughput bump from `model = model.to(memory_format=torch.channels_last)`.

*Diagnostic angle for TraceML.* If a layer's GPU time is suspicious (e.g., a conv that should be Tensor-Core-friendly is much slower than expected), one suspect is layout: input is in NCHW-contiguous but the kernel wants channels-last. A future TraceML feature could detect "input arrived in stride X, kernel wants stride Y, here's the implied copy cost." Today, the user sees only the aggregate slowdown.

*Mental model.* Contiguity is to tensors what cache-line-friendliness is to CPU code: a property of memory layout that determines whether the hardware can run at peak speed or has to chase pointers. Most of the time you don't have to think about it because PyTorch handles it; when you do have to think about it, it's because something is silently allocating, silently slow, or noisily refusing.

**Concepts introduced:** contiguous strides formula, `.is_contiguous()` invariants, row-major (C-order) layout, sources of non-contiguity (transpose/permute/expand/strided slice), `view` (strict) vs `reshape` (with fallback copy), coalesced memory access on GPU, kernel layout assumptions, `.contiguous()` cost model, hidden copies as a perf footgun, `channels_last` memory format and Tensor Core layouts, layout-aware bottleneck diagnosis.

---

## nn.Module mechanics

### P7: What is `nn.Module` exactly, and why is `__call__` different from `forward`?

**Date:** 2026-04-24

**Short answer:** `nn.Module` is the base class for **stateful neural network components** in PyTorch. It tracks **parameters**, **buffers**, and **child modules** via overridden `__setattr__`, exposes utilities (`to`, `state_dict`, `parameters`), and routes calls through `__call__`. You define behavior in `forward`, but you invoke the module as `model(x)` because `__call__` is a wrapper that fires **hooks** and only then dispatches to `forward`. Calling `forward` directly bypasses all of that.

**Long answer:**

*What `nn.Module` actually holds.* An `nn.Module` instance has a handful of dict attributes maintained by overridden `__setattr__`:

- `self._parameters: Dict[str, Parameter]` — learnable tensors (see [P8](#p8-whats-the-difference-between-a-parameter-a-buffer-and-a-plain-tensor-attribute)).
- `self._buffers: Dict[str, Tensor]` — non-learnable persistent state.
- `self._modules: Dict[str, Module]` — child submodules.
- `self._forward_hooks`, `self._forward_pre_hooks`, etc. — `OrderedDict`s of registered callbacks.
- `self.training: bool` — mode flag (see [P10](#p10-what-does-setting-an-nnmodule-to-training-vs-evaluation-mode-actually-change-at-runtime-batchnorm-dropout-autograd-state)).

When you write `self.linear = nn.Linear(10, 10)` inside `__init__`, the overridden `__setattr__` notices the value is an `nn.Module` and routes it into `self._modules["linear"]`. This `__setattr__` magic is the single most important mechanic in `nn.Module`: it's how the module "sees" its own parameters and children.

*The `__call__` vs `forward` split.* `forward` is what *you* write — pure tensor math. But you call it as `out = model(x)`. That goes through `nn.Module.__call__` (internally `_call_impl`), which fires forward pre-hooks, calls `self.forward(...)`, fires forward hooks, registers backward hooks on outputs, and returns. This is why TraceML can attach `register_forward_hook` and observe activations without you ever calling a TraceML function — the hook firing is built into `__call__`, not `forward`.

*Why this matters.*

- Calling `model.forward(x)` skips every hook and hook-driven instrumentation.
- `torch.compile` traces through `__call__`, not bare `forward`.
- TraceML's `_call_impl` patching gives finer control over timing than relying solely on public hooks (see [Q9](learning-qa.md#q9-what-are-hooks-and-what-does-injecting-hooks-in-process-mean) and [P48](#p48-what-is-_call_impl-and-why-does-traceml-monkey-patch-around-it-instead-of-just-using-public-hooks)).

*Mental model.* `nn.Module` is a "smart container" + "callable wrapper": tracks parameters/buffers/children automatically, and `__call__` is the published entry point with `forward` as the user-overridable kernel. Hooks fire in `__call__`, never in bare `forward`.

**Concepts introduced:** `nn.Module` base class, overridden `__setattr__`, `_parameters`/`_buffers`/`_modules` dicts, `__call__` vs `forward`, `_call_impl`, hook firing in `__call__`, why never call `forward` directly.

---

### P8: What's the difference between a `Parameter`, a `Buffer`, and a plain tensor attribute?

**Date:** 2026-04-24

**Short answer:** A **`Parameter`** is a `torch.Tensor` subclass that, when assigned to an `nn.Module`, automatically gets registered in `self._parameters`, included in `model.parameters()` (so optimizers update it), saved in `state_dict()`, and moved by `model.to(device)`. A **buffer** is a regular tensor explicitly registered via `self.register_buffer("name", tensor)` — it's persistent state that `state_dict()` saves and `.to()` moves, but the optimizer does **not** update it. A **plain tensor attribute** (`self.foo = torch.zeros(10)` without registration) is invisible to all of this — not in `state_dict`, not moved by `.to`, not seen by the optimizer.

**Long answer:**

*The three categories, side by side.*

|                        | `nn.Parameter`     | Buffer                 | Plain tensor attribute |
| ---------------------- | ------------------ | ---------------------- | ---------------------- |
| In `parameters()`?     | Yes                | No                     | No                     |
| In `buffers()`?        | No                 | Yes                    | No                     |
| In `state_dict()`?     | Yes                | Yes (if persistent)    | No                     |
| Moved by `.to(device)` | Yes                | Yes                    | No                     |
| Optimizer updates it?  | Yes                | No                     | No                     |
| `requires_grad` default| `True`             | `False`                | as created             |
| Typical use            | weights, biases    | running mean/var, masks| scratch values         |

*Why `Parameter` is a Tensor subclass.* `nn.Parameter` is essentially:

```python
class Parameter(torch.Tensor):
    def __new__(cls, data=None, requires_grad=True):
        ...  # wraps a tensor, sets requires_grad=True by default
```

It's a **marker class** — the only reason it exists is so `nn.Module.__setattr__` can do `isinstance(value, Parameter)` and route it into `self._parameters`. The tensor math is identical.

*Buffers: persistent non-learnable state.* `BatchNorm` is the canonical example. It needs `running_mean` / `running_var` to be saved with the model, moved with `.to`, but never updated by SGD. So inside `BatchNorm.__init__`:

```python
self.register_buffer("running_mean", torch.zeros(num_features))
self.register_buffer("running_var", torch.ones(num_features))
```

The `persistent=False` flag (e.g., for precomputed attention masks you can rebuild) makes a buffer skip serialization but still get device-migrated.

*Plain attribute footgun.* If you write `self.mask = torch.ones(10)` without `register_buffer`, `model.cuda()` leaves `self.mask` on CPU — next forward crashes with a device mismatch. Same trap with `state_dict`: a plain attribute won't be saved.

*Why this matters for TraceML.* TraceML's per-layer memory sampler walks `module.parameters()` and `module.buffers()` to size each layer's resident memory. Plain attributes are invisible — sampler undercounts memory. A user noticing "TraceML's reported model size is smaller than my checkpoint" should suspect unregistered tensors.

**Concepts introduced:** `nn.Parameter` Tensor subclass, marker class pattern, `register_buffer`, persistent vs non-persistent buffers, `__setattr__` dispatch into `_parameters`/`_buffers`/`_modules`, plain-attribute footgun (device mismatch, missing-from-checkpoint).

---

### P9: How does `state_dict()` work, and what does it preserve / what does it skip?

**Date:** 2026-04-24

**Short answer:** `state_dict()` returns an `OrderedDict[str, Tensor]` mapping dotted parameter/buffer paths (e.g., `"layer1.0.weight"`) to their **tensor values**. It walks the module tree recursively and collects everything in `self._parameters` and `self._buffers` (the latter only if `persistent=True`). It is essentially a flat snapshot of learnable + persistent state, keyed by structural position. It does **not** preserve module classes, code, optimizer state, hooks, RNG state, training-mode flag, or plain Python attributes.

**Long answer:**

*Mechanics.* When you call `model.state_dict()`, internally it walks the tree depth-first: for each `(name, param)` in `self._parameters.items()`, insert `f"{prefix}{name}" -> param.detach()`. Same for persistent buffers. Recurse into `self._modules.items()` with prefix appended.

For:

```python
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        self.bn = nn.BatchNorm1d(5)
```

You get:

```
['encoder.0.weight', 'encoder.0.bias',
 'encoder.2.weight', 'encoder.2.bias',
 'bn.weight', 'bn.bias',
 'bn.running_mean', 'bn.running_var', 'bn.num_batches_tracked']
```

`encoder.1` (the `ReLU`) doesn't appear — no parameters or buffers. BatchNorm contributes both parameters (`weight`, `bias`) and buffers (`running_mean`, `running_var`, `num_batches_tracked`).

*What it does NOT preserve.*

- **Module classes / code.** A state dict has no idea whether `encoder.0` is a `Linear`. You need the matching architecture instance on the load side.
- **Non-persistent buffers** (`persistent=False`).
- **Plain tensor attributes** (per [P8](#p8-whats-the-difference-between-a-parameter-a-buffer-and-a-plain-tensor-attribute)).
- **Optimizer state** — separate via `optimizer.state_dict()`.
- **Hooks** — re-attach after `load_state_dict`.
- **RNG state** — use `torch.get_rng_state()` / `torch.cuda.get_rng_state_all()` separately.
- **Training-mode flag** (`self.training`). Loaded model needs explicit `model.eval` or `model.train` after loading. See [P10](#p10-what-does-setting-an-nnmodule-to-training-vs-evaluation-mode-actually-change-at-runtime-batchnorm-dropout-autograd-state).
- **Custom Python state** (lists, dicts, hyperparameters as floats).

*Loading: `model.load_state_dict(sd, strict=True)`.* Walks the tree, navigates to each parameter/buffer slot, and does `target.data.copy_(tensor)` — an **in-place copy**, not reassignment. Why: optimizers hold references to the original `Parameter` objects from `model.parameters()` at construction. If `load_state_dict` rebound them, the optimizer would point at stale tensors. With `strict=False` you can do partial loads (fine-tuning a backbone with a fresh head).

*Customization hooks.* `_register_state_dict_hook`, `_register_load_state_dict_pre_hook` let frameworks (Lightning, HuggingFace) inject extra processing for renamed keys across model versions.

*Mental model.* `state_dict()` is a **structural diff between architecture and weights**: the architecture is code (in `__init__` and `forward`), the weights are values. Saving = saving the values; loading = re-running the code, then injecting values. Two slightly different architectures can share weights for compatible submodules with `strict=False`; a saved checkpoint is useless without the code that produced its keys.

**Concepts introduced:** `state_dict()` recursive collection, dotted-path keys, persistent vs non-persistent buffers, what's *not* in `state_dict`, `load_state_dict` in-place `.copy_` semantics, `strict=True` matching contract, `strict=False` partial load, `_register_state_dict_hook` extension points.

---

### P10: What does setting an `nn.Module` to training vs evaluation mode actually change at runtime (BatchNorm, Dropout, autograd state)?

**Date:** 2026-04-24

**Short answer:** Toggling mode just flips a boolean flag — `self.training = True` or `False` — recursively across the module tree. The mode toggle methods on `nn.Module` are trivial setters: the training-mode method takes a `mode: bool` argument, sets `self.training`, and recursively calls itself on every child; the evaluation-mode method just calls the former with `False`. Almost no PyTorch code reads the flag. The ones that do are layers whose behavior **must** differ between fitting and inference: `BatchNorm`/`SyncBatchNorm` (training mode updates running statistics; evaluation mode uses them) and `Dropout` (training mode masks activations; evaluation mode is the identity). The mode flag has **no effect on autograd** — gradient tracking is controlled separately by `torch.no_grad`/`torch.inference_mode`/`requires_grad`.

**Long answer:**

*What the mode toggle methods actually do.* Both are recursive setters for one boolean. Toggling mode **doesn't reconfigure layers, doesn't move tensors, doesn't change autograd state**. It just flips `self.training` everywhere in the tree. The behavioral difference comes from the handful of layers that read the flag inside `forward`.

In code, you call `model.train` (with parens at runtime; shown bare here) before your training loop and `model.eval` before validation, typically pairing the latter with `torch.no_grad()` to also disable autograd.

*Layers that read `self.training`.*

- **`Dropout` family.** Training mode → samples a Bernoulli mask, multiplies activations by `mask / (1-p)` (inverted dropout). Evaluation mode → identity. Forgetting to switch to evaluation mode means your validation loss has noisy activations.
- **`BatchNorm` family.** Training mode → compute batch mean/variance from the current minibatch, normalize, **update** `running_mean`/`running_var` via EMA, increment `num_batches_tracked`. Evaluation mode → use frozen running stats; don't touch them. Forgetting eval-mode during validation pollutes the running stats with validation data — a classic catastrophe.
- **`LayerNorm`, `GroupNorm`, `InstanceNorm`** — mostly mode-independent (no running stats by default).
- **`RNN`/`LSTM`/`GRU`** with `dropout > 0` — dropout between layers respects `self.training`.
- **Custom layers** that branch on `self.training` (e.g., Stochastic Depth).

*The flag does NOT control autograd.* This is the most common confusion. Switching to evaluation mode **does not disable gradient computation**. Forward still builds the autograd graph; activations are still saved for backward; memory usage is identical to training-mode forward. To stop autograd you need `torch.no_grad()` or `torch.inference_mode()` — orthogonal to the mode flag.

*Bug pattern.*

```python
for batch in val_loader:
    # forgot the eval-mode switch — Dropout still drops, BN updates running stats
    with torch.no_grad():  # at least we saved the memory
        val_loss = model(batch).mean()
```

The `no_grad` block saves you on memory but does not save you on BN/Dropout correctness. You need *both* the eval-mode switch and `torch.no_grad()` (or `inference_mode`).

*Freezing BatchNorm during fine-tuning.* If you want BN to use its pre-trained running stats while the rest of the model trains, switch BN submodules to eval-mode individually:

```python
def freeze_bn(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval         # this flips m.training=False for just this submodule

model.train         # whole model in training mode
model.apply(freeze_bn)  # then individually evaluation-mode the BN layers
```

Works because the mode flag is per-module, and `forward` consults the flag of the BN module itself.

*Why this matters for TraceML.* TraceML attaches forward hooks regardless of mode. But the *cost* of a forward changes with mode — eval-mode forward through a model with no `no_grad` wrap still allocates activations for backward. So if a user reports "memory during validation is the same as training," the diagnosis is usually "forgot the `no_grad` wrap," not anything mode-related.

**Concepts introduced:** mode flag (`self.training`) as a recursive boolean, mode-toggle methods being trivial setters not reconfigurations, `Dropout` mode-dependence, `BatchNorm` mode-dependence (running-stat updates and the validation pollution bug), normalization layers without running stats, orthogonality of mode flag vs autograd controls, per-module mode application (BN-freeze idiom for fine-tuning).

---

### P11: What does `model.to(device)` do as it traverses the module tree?

**Date:** 2026-04-24

**Short answer:** `model.to(device)` walks the module tree depth-first and calls `.to(device)` on every parameter and buffer it finds (via the `_apply` helper), **mutating the module in place** (returns `self` for chaining). Each individual parameter/buffer move is a tensor `.to(device)` — synchronous on the default stream unless the source is pinned host memory and `non_blocking=True` is requested. Plain tensor attributes are not touched. The method also handles `dtype` and memory format.

**Long answer:**

*The traversal mechanism: `_apply`.* Under the hood, `nn.Module.to` is a thin wrapper that builds a per-tensor conversion function and hands it to `self._apply(fn)`. `_apply` recurses through `self.children()`, then for each parameter wraps the moved tensor in a fresh `Parameter` and reassigns into `self._parameters[key]`. Same for `self._buffers`. Returns `self`.

Three things to notice:

1. **Depth-first recursion** through `self.children()`. Leaves move before the root.
2. **Only `_parameters` and `_buffers` are touched.** Plain attributes are invisible — source of the device-mismatch footgun (see [P8](#p8-whats-the-difference-between-a-parameter-a-buffer-and-a-plain-tensor-attribute)).
3. **`self._parameters[key]` is reassigned** with a fresh `Parameter` wrapping the moved data. The optimizer holds references to the original parameters from `model.parameters()` at construction — which is why **you should always create the optimizer after `model.to(device)`**. If you do it the other way, the optimizer iterates over CPU `Parameter` objects, and `optimizer.step()` updates ghost tensors on CPU while your model trains on GPU — silent failure.

*The per-tensor `.to(device)` semantics.* See [P3](#p3-what-does-todevice-actually-do-under-the-hood-and-is-the-copy-synchronous-or-asynchronous): default-synchronous on CPU↔GPU, async only with pinned source + `non_blocking=True`.

*What `model.to(...)` accepts.*

```python
model.to("cuda")
model.to("cuda:1")
model.to(torch.bfloat16)                        # dtype conversion
model.to("cuda", torch.bfloat16)                # device + dtype
model.to(memory_format=torch.channels_last)     # for vision models
model.to(another_tensor)                         # match device + dtype of another_tensor
```

*Mutation asymmetry: modules vs tensors.* For modules, `.to` mutates in place but returns `self` for chaining. For tensors, `.to` returns a new tensor (no container to rebind into):

```python
t = torch.zeros(10)
t.to("cuda")     # WRONG — discards the result
t = t.to("cuda") # correct
```

Modules: assign-or-not is fine. Tensors: must reassign.

*Effect on `.grad`.* If a parameter has a `.grad` tensor, `_apply` moves the gradient too. The optimizer's view of `param.grad` follows the move.

*Why this matters for TraceML.* TraceML's layer-memory and layer-time samplers walk the same module tree to attach hooks. The API guidance is "call `trace_model_instance` after `model.to(device)`" — same advice as for the optimizer, for the same structural reason.

**Concepts introduced:** `nn.Module._apply` traversal, depth-first recursion through `_modules`, in-place reassignment of `_parameters`/`_buffers`, why optimizer construction order matters, per-tensor `.to(device)` semantics, `.to` polymorphism (device/dtype/memory format), module-vs-tensor mutation asymmetry, gradient tensors moved alongside parameters.

---

### P12: What's the difference between `children()`, `modules()`, `named_modules()`?

**Date:** 2026-04-24

**Short answer:** `children()` yields **direct submodules only** (one level deep). `modules()` yields **every module in the tree, including `self`**, depth-first, deduplicated by identity. `named_modules()` does the same as `modules()` but yields `(dotted_name, module)` pairs — the dotted name is the same path scheme `state_dict()` keys use. Use `children()` for shallow per-block iteration; use `modules()` / `named_modules()` for full recursive traversal.

**Long answer:**

*A worked example.*

```python
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block()
        self.block2 = Block()
        self.head = nn.Linear(10, 5)

net = Net()
```

- `list(net.children())` → `[Block, Block, Linear]` — three top-level items. Shallow iterator over `self._modules.values()`.
- `list(net.modules())` → 8 items: `Net`, then `Block`, `Linear (fc)`, `BatchNorm1d (bn)` for each block, then `Linear (head)`. Depth-first preorder. Deduplicated by `id()` — tied weights yield the shared module once.
- `list(net.named_modules())` → same as `modules()` but with dotted names: `("", Net)`, `("block1", Block)`, `("block1.fc", Linear)`, `("block1.bn", BatchNorm1d)`, etc. The empty-string name is the root. **This is the iterator TraceML uses for per-layer hook attachment** — the dotted name becomes the layer's identifier in the dashboard.

*Cousins.*

- `parameters()` / `named_parameters()` — flat iteration over all `Parameter` instances.
- `buffers()` / `named_buffers()` — same for buffers.
- `apply(fn)` — calls `fn(submodule)` on every module in the tree. Common idiom for weight init.

*When to use which.*

| Scenario | Pick |
| --- | --- |
| Wrap each top-level block (FSDP unit boundary) | `named_children()` |
| Attach a forward hook to every layer | `named_modules()` |
| Find all `BatchNorm` modules to freeze | `modules()` + isinstance |
| Get all learnable tensors for the optimizer | `parameters()` |
| Save/load state | `state_dict()` |
| Run weight init | `apply(fn)` |

*Container modules.* `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict` are themselves `nn.Module`s whose children are stored in `_modules` with integer-like keys. So `state_dict` keys for `nn.Sequential` look like `encoder.0.weight`. A regular Python `list` of modules is **invisible** to all these iterators — same family of bug as plain tensor attributes. Use `nn.ModuleList` instead.

*Why this matters for TraceML.* The layer-time and layer-memory samplers iterate `model.named_modules()` to attach per-layer hooks. The dotted name shows up in the dashboard as the layer label. Plain Python lists of modules → invisible to TraceML — same diagnostic flag as "missing from `state_dict`": broken `nn.Module` registration.

**Concepts introduced:** `children()` (direct only), `modules()` (recursive + dedup), `named_modules()` (recursive with dotted names matching `state_dict` keys), parallel `parameters()`/`buffers()` iterators, `apply(fn)` recursive function application, container modules (`Sequential`/`ModuleList`/`ModuleDict`), plain Python list as a footgun.

---

## Autograd

### P13: Is PyTorch's computation graph static or dynamic? What does that imply for instrumentation?

**Date:** 2026-04-24

**Short answer:** PyTorch's autograd graph is **dynamic** (a.k.a. **define-by-run**): it is constructed on-the-fly during the forward pass as a side effect of executing tensor ops, and discarded after `.backward()`. Each iteration builds a fresh graph. The implication for instrumentation is huge: you cannot inspect "the graph" ahead of time — you must observe it as it is built, which is exactly why PyTorch exposes runtime hooks rather than a graph-rewriting API.

**Long answer:**

*Define-by-run.* When you write:

```python
x = torch.randn(4, 8, requires_grad=True)
y = x @ W           # MatmulBackward node attached to y
z = torch.relu(y)   # ReluBackward node attached to z
loss = z.sum()      # SumBackward node attached to loss
loss.backward()     # walks the graph backwards
```

each tensor op does two things: (1) computes the output value, and (2) appends a node to a DAG that records *what produced this tensor and from which inputs*. The graph lives in the tensors themselves: each output tensor carries a `grad_fn` pointer to the backward node, and each backward node carries `next_functions` pointers to its inputs' `grad_fn`s. There is no global "graph object" — the graph is the chain of `grad_fn`s reachable from `loss`.

*Why "dynamic" matters.* You can use Python control flow (`if`, `for`, `while`) inside forward — different iterations produce different graph structures (variable-depth RNNs, MoE routing, REINFORCE-style sampling). You can print, debug, breakpoint, mid-forward — tensors are real Python objects. You can mutate model architecture between steps. The cost: every step pays for graph construction and teardown. Static frameworks amortize this; PyTorch traded it for ergonomics, and `torch.compile` is the modern attempt to claw some of it back.

*Implications for instrumentation.*

1. **No ahead-of-time analysis.** There *is* no graph until you run forward. Tools that need a full graph either trace one forward (`torch.fx.symbolic_trace`, `torch.jit.trace`) or intercept Python bytecode at runtime (`torch.compile`/TorchDynamo).
2. **Instrumentation must hook into execution itself.** Module hooks fire when an `nn.Module`'s `__call__` runs; tensor hooks fire when a tensor's gradient is computed; autograd profiler wraps op dispatch. TraceML uses the first two for layer-level timing/memory.
3. **Per-step overhead, not per-program.** Anything you do during the build (extra Python in a hook, an extra event recording) gets paid once per step. TraceML's design — append-only deques, CUDA event reuse pools, batched TCP sends — is a direct response.

*RL angle.* In RL the graph genuinely changes shape: episode length varies, advantage windows grow, exploration branches diverge. Static-graph frameworks would force you to pad-and-mask everything (TF1 RL code from 2018 was full of this). Dynamic graphs let you write the math the way you'd write the equations.

*Mental model.* The graph is not a blueprint — it's an audit trail. Every op leaves a breadcrumb (`grad_fn`); `.backward()` walks them in reverse; the breadcrumbs vanish after one walk. To watch the trail being laid, you walk alongside the runner, not study the map afterwards.

**Concepts introduced:** dynamic computation graph, define-by-run, `grad_fn`, `next_functions`, autograd engine as runtime walker, tracing vs eager mode, TorchDynamo as dynamic-to-static bridge, instrumentation-must-be-runtime corollary, per-step instrumentation overhead budget.

---

### P14: What does `loss.backward()` actually do, step by step?

**Date:** 2026-04-24

**Short answer:** `loss.backward()` hands the autograd graph rooted at `loss` to the **autograd engine** (a C++ component with worker threads), which performs a **reverse topological traversal**: starting from `loss` with an implicit gradient of `1.0`, it visits each `grad_fn`, runs its backward to compute input gradients via the chain rule, accumulates gradients into the `.grad` field of every leaf tensor it reaches, and frees the graph (unless `retain_graph=True`). On GPU, the actual gradient kernels are queued onto CUDA streams and execute asynchronously — `loss.backward()` returns when the *queueing* is done, not when the GPU has finished.

**Long answer:**

*The graph going in.* Recall from [P13](#p13-is-pytorchs-computation-graph-static-or-dynamic-what-does-that-imply-for-instrumentation): each non-leaf tensor carries a `grad_fn` pointing to a backward node; each backward node carries `next_functions` linking to inputs' `grad_fn`s. The full graph reachable from `loss.grad_fn` is the autograd graph.

*Steps the engine performs.*

1. **Validation and seeding.** Check `loss` is scalar (or you provided `gradient=`). Build the initial gradient `torch.ones_like(loss)`, i.e., `dL/dL = 1`.
2. **Hand off to C++.** Python calls into `torch.autograd.backward(...)` → `torch::autograd::Engine::execute(...)`. The GIL is released while waiting on the engine.
3. **Build dependency map.** Engine walks the graph from `loss.grad_fn`, counts incoming edges per node. A node is "ready" when all its dependents have run.
4. **Schedule onto worker threads.** A small pool — typically one per CUDA device + one for CPU. Each worker has a ready queue. Workers pop nodes, execute, push newly-ready downstream nodes.
5. **Execute one node.** For e.g. `MmBackward` for `Y = X @ W`: gather incoming gradient(s) `grad_Y`; call `apply()` → `grad_X = grad_Y @ W.T`, `grad_W = X.T @ grad_Y`. On GPU these ops are queued to streams and return immediately. Hand resulting gradients to `next_functions`.
6. **Hooks fire.** Tensor hooks (`tensor.register_hook(fn)`) fire as the engine produces a gradient for that tensor. Module backward hooks fire when a module's backward as a whole completes. Hooks fire on autograd worker threads — matters for thread-safety.
7. **Graph teardown.** After traversal, saved tensors held by each `grad_fn` are released. `retain_graph=True` skips this (see [P17](#p17-when-do-you-need-retain_graphtrue)).
8. **Return to Python — but the GPU is still running.** `loss.backward()` returns when CPU-side bookkeeping is done and kernels are queued. On GPU workloads, the actual gradient compute is in flight on streams. `param.grad.cpu()` would force a sync. This is why "wrap `loss.backward()` with `time.perf_counter`" only measures host-side backward, not GPU-side completion.

*Why TraceML uses CUDA events for backward timing.* Host-time measures dispatch/queueing. GPU-time (via CUDA events on the relevant stream — see [Q15](learning-qa.md#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread)) measures actual completion. TraceML records both to decompose backward into "Python/dispatch overhead" vs "true GPU compute time."

*Numerical accumulation detail.* When the same parameter is reached through multiple paths (shared embedding used twice), `AccumulateGrad` sums contributions before they land in `.grad`. This is the chain rule's sum-over-paths, the same mechanism used deliberately for gradient accumulation across microbatches ([P16](#p16-why-do-gradients-accumulate-by-default-when-is-that-useful-when-is-it-a-footgun)).

*Mental model.* `.backward()` is a job submission. You hand the engine a DAG and say "compute leaf gradients via chain rule, accumulate into `.grad`." The engine schedules nodes in topo order across worker threads. On GPU it's mostly queueing; the GPU drains the queue after you've returned.

**Concepts introduced:** autograd engine (C++), reverse-topological traversal, dependency-counting / ready queue, per-device worker threads, AccumulateGrad node, saved tensors and graph teardown, hook firing on worker threads, async backward kernel queueing, host-time vs GPU-time decomposition.

---

### P15: What's the difference between `.grad`, `.grad_fn`, and the autograd engine's task graph?

**Date:** 2026-04-24

**Short answer:** Three different objects at three levels. **`.grad`** is a tensor attribute holding the *accumulated gradient value* for a leaf tensor — populated by `.backward()`. **`.grad_fn`** is a pointer on a non-leaf tensor to the *backward function node* that knows how to compute upstream gradients from this tensor's gradient — set by autograd at forward-op time. The **autograd engine's task graph** is the *runtime DAG of backward nodes* the engine schedules during `.backward()` — built from the linked `grad_fn` chain plus dependency counts plus per-device worker queues.

**Long answer:**

*`.grad` — the bucket where leaf gradients land.*

```python
W = torch.randn(8, 4, requires_grad=True)
print(W.grad)        # None — no backward has run yet
loss = (W ** 2).sum()
loss.backward()
print(W.grad.shape)  # torch.Size([8, 4])
```

Set on **leaf tensors** only (you can use `tensor.retain_grad()` to populate it for non-leaves). Writes accumulate (`+=`, see [P16](#p16-why-do-gradients-accumulate-by-default-when-is-that-useful-when-is-it-a-footgun)). What the optimizer reads in `optimizer.step()`.

*`.grad_fn` — the per-tensor breadcrumb.*

```python
x = torch.randn(4, requires_grad=True)
y = x * 2            # y.grad_fn = <MulBackward0>
z = y.sin()          # z.grad_fn = <SinBackward0>
print(z.grad_fn.next_functions)   # ((<MulBackward0>, 0),)
```

Every non-leaf tensor produced by a differentiable op gets a `grad_fn`. The grad_fn:
- Stores any tensors needed for backward (e.g., `MulBackward0` saves the *other* operand).
- Holds `next_functions` — pointers to inputs' `grad_fn`s.
- Implements `apply(grad)` — the math turning output gradient into input gradients.

The chain `z.grad_fn → y.grad_fn → x.grad_fn` *is* the autograd graph. No separate object. Leaves don't have a `grad_fn`; the traversal terminates at `AccumulateGrad` nodes that write into leaves' `.grad`.

*The engine's task graph.* When you call `.backward()`, the engine reads the linked `grad_fn` chain and constructs internal scheduling state:

- **Dependency counts** per node.
- **Per-device ready queues**.
- **Worker threads** popping ready nodes.
- **Saved tensor lifetimes** — released after a node executes (unless `retain_graph=True`).

This is the runtime task graph: same nodes as the static `grad_fn` chain, enriched with dependency counts, ready-state, queue assignment.

*Why the three-way split matters for tooling.*

- TraceML's per-layer timing lives at the **module** level (forward/backward hooks).
- Per-tensor gradient inspection lives at the `.grad` level (post-backward) or via `register_hook` (during backward).
- TraceML does *not* walk the engine's task graph or modify scheduler internals — that layer is private C++.

*Common confusions.*

- "The autograd graph" can mean any of the three. The `grad_fn` chain is the *structure*; the engine's task graph is the *scheduled execution*; `.grad` is the *output*.
- `.grad_fn` is `None` on leaves and on tensors with `requires_grad=False`. Also `None` after `detach()` (see [P18](#p18-whats-the-difference-between-detach-no_grad-inference_mode-and-requires_gradfalse)).
- A tensor can have *both* `.grad` and a populated forward graph reaching it via `AccumulateGrad`. They coexist.

**Concepts introduced:** `.grad` as accumulator on leaves, `.grad_fn` as backward-function pointer, `next_functions` chain, `AccumulateGrad` terminator node, leaf vs non-leaf, `retain_grad` for intermediate gradients, runtime task graph, dependency counting, per-device ready queues.

---

### P16: Why do gradients accumulate by default? When is that useful, when is it a footgun?

**Date:** 2026-04-24

**Short answer:** Gradients accumulate (`grad += new_grad`) because the chain rule itself is a sum-over-paths: when a parameter is reached by multiple gradient paths, those contributions must add. PyTorch exposes the same mechanism across multiple `.backward()` calls, enabling **gradient accumulation** (large effective batch size on small memory) and **multi-loss training** for free. The footgun: forgetting to zero gradients between training steps means `.grad` silently keeps summing, effective LR doubles every step, loss explodes — with no error message.

**Long answer:**

*Why accumulation is mathematically necessary.* For a shared parameter `W` used in two paths:

```python
y1 = x1 @ W
y2 = x2 @ W
loss = y1.sum() + y2.sum()
loss.backward()
```

Chain rule gives `dL/dW = x1.T @ ones + x2.T @ ones`. Engine walks both paths and writes both contributions into `W.grad` — must be `+=`. Implemented by `AccumulateGrad` (see [P15](#p15-whats-the-difference-between-grad-grad_fn-and-the-autograd-engines-task-graph)).

*Useful cases.*

**1. Microbatching.** Train with effective batch 256 on memory for 32:

```python
optimizer.zero_grad()
for microbatch in chunk(batch, n=8):
    loss = model(microbatch).mean() / 8
    loss.backward()                  # accumulates into .grad
optimizer.step()
optimizer.zero_grad()
```

After the loop, every parameter's `.grad` is the average gradient across all 8 microbatches — equivalent to one backward on full batch 256 (modulo BatchNorm).

**2. Multi-loss training without summing tensors.** Calling `.backward()` on each loss accumulates into `.grad`, equivalent to `(L1 + L2 + ...).backward()`. Cleaner code, sometimes lower peak memory.

**3. RL patterns.** Off-policy methods with multiple gradient terms (policy, value, entropy bonus) sometimes naturally fall out as separate `.backward()` calls.

*The footgun.* Without `optimizer.zero_grad()`:

- Step 1: `.grad` = g1. Step takes g1.
- Step 2: `.grad` = g1 + g2. Step takes g1 + g2 — **double-counting g1**.
- Step 3: `.grad` = g1 + g2 + g3 — and so on.

Effective LR grows every step. Loss explodes within a few iterations. No warning, no exception. The framework can't tell whether you forgot or you wanted accumulation. Most common training bug for newcomers.

*`zero_grad()` vs `zero_grad(set_to_none=True)`.* Modern default is `set_to_none=True`:

- Cheaper — no kernel launch to fill zeros.
- Next backward sees `.grad is None` and *creates* a fresh tensor, one fewer add.
- Optimizers must handle `param.grad is None` correctly. With `set_to_none=False`, parameters with no gradient signal got an update step against zero (usually a no-op, but with weight decay it wasn't).

*DDP interaction.* With gradient accumulation across microbatches, you usually want to *skip* gradient sync until the last microbatch:

```python
for i, microbatch in enumerate(chunks):
    sync_context = (model.no_sync() if i < len(chunks) - 1
                    else contextlib.nullcontext())
    with sync_context:
        loss = model(microbatch).mean() / len(chunks)
        loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

`model.no_sync()` skips the all-reduce on this backward, accumulates locally; on the final microbatch, all-reduce sums all accumulated gradients across ranks.

*TraceML angle.* TraceML's "step boundary" is `trace_step()`, not `.backward()`. A microbatched step (8 backwards, 1 optimizer step) is one TraceML step — matches the user's mental model.

**Concepts introduced:** chain-rule sum-over-paths, AccumulateGrad as `+=`, gradient accumulation pattern (microbatching), multi-loss accumulation, no-zero-grad footgun, `set_to_none=True` semantics, DDP `no_sync()` for accumulation, TraceML step-vs-backward distinction.

---

### P17: When do you need `retain_graph=True`?

**Date:** 2026-04-24

**Short answer:** You need `retain_graph=True` when you want to call `.backward()` (or `torch.autograd.grad`) **more than once on the same forward pass**. By default PyTorch frees saved tensors held by `grad_fn` nodes immediately after backward — those saved tensors are needed to recompute gradients, so freeing them invalidates the graph. Common cases: **second-order gradients** (gradient of a gradient — MAML, influence functions, WGAN-GP), **two losses with shared subgraph as separate backwards**, or **GAN-like setups**. It's a memory-correctness escape hatch, not a performance feature — keeping the graph means keeping every saved activation alive.

**Long answer:**

*Why backward frees the graph by default.* Backward nodes hold **saved tensors** — copies of forward-pass tensors needed to compute the local gradient. For `MmBackward` of `Y = X @ W`: saves `X` and `W` to compute `grad_X = grad_Y @ W.T` and `grad_W = X.T @ grad_Y`. These dominate "activation memory."

After backward visits a node and produces input gradients, saved tensors are released. This is why memory drops dramatically right after `.backward()` returns. Try a second backward without `retain_graph=True`:

```python
loss.backward()
loss.backward()   # RuntimeError: Trying to backward through the graph a second time...
```

*When you actually need it.*

**1. Higher-order gradients.** To compute `d²L/dW²` you backprop *through* the first backward — meaning the first backward's computation graph must itself remain differentiable.

```python
loss = model(x).sum()
grad_W = torch.autograd.grad(loss, model.W, create_graph=True)[0]
hessian_diag = torch.autograd.grad(grad_W.sum(), model.W)[0]
```

`create_graph=True` is the higher-order sibling — both retains and builds a new graph over the first backward's ops. Used in MAML (inner-loop SGD step is differentiated through), influence functions, implicit differentiation, WGAN-GP.

**2. Multiple losses on shared subgraph, separately.** If you backprop two losses through a shared encoder one at a time (e.g., for per-loss gradient logging), the first backward needs `retain_graph=True` so the encoder's saved tensors survive for the second.

**3. GAN-style multi-stage backward** (sometimes).

*Cost.* `retain_graph=True` keeps all saved tensors alive until the original forward result is GC'd. For a deep transformer, can double or triple peak memory. People reach for it when they can't avoid it.

**Gradient checkpointing** (`torch.utils.checkpoint.checkpoint`) is the opposite tradeoff: don't save activations during forward, recompute them during backward.

*Common confusions.*

- `retain_graph=True` does **not** mean "keep the graph for the next training step." Each forward builds a fresh graph regardless.
- `retain_graph=True` ≠ `create_graph=True`. The former keeps saved tensors so you can backprop again; the latter additionally makes the backward computation differentiable.

*Footgun: silent retention.* `losses.append(loss)` instead of `losses.append(loss.item())` keeps the entire forward graph alive forever. Memory grows linearly with iterations. Always extract scalar values via `.item()` for logging.

*TraceML angle.* `retain_graph=True` every step → much higher peak memory. TraceML's step-memory sampler reports it; large peak with growth across steps suggests accidentally retained loss tensors.

**Concepts introduced:** saved tensors, activation memory, default graph teardown, retain_graph mechanics, create_graph for higher-order gradients, MAML/influence/WGAN-GP as canonical second-order use cases, gradient checkpointing as opposite tradeoff, accidental graph retention via stored loss tensors.

---

### P18: What's the difference between `detach()`, `no_grad()`, `inference_mode()`, and `requires_grad=False`?

**Date:** 2026-04-24

**Short answer:** All four prevent autograd from tracking ops, at different scopes and costs. **`requires_grad=False`** is a *per-tensor flag*: ops involving only such tensors don't build graph. **`detach()`** returns a *new tensor sharing storage* with the original but disconnected from the graph — surgical mid-forward cut. **`torch.no_grad()`** is a *context manager* disabling graph building globally for any op in the block. **`torch.inference_mode()`** is a stricter, faster successor to `no_grad()` that additionally disables version counters and view tracking — for serving where you'll never call backward.

**Long answer:**

*The shared goal.* Skip graph construction. Each differentiable op allocates a `grad_fn` node, links it to inputs, possibly saves tensors. Wasted memory and a bit of compute when you don't need gradients (inference, target networks in DQN, EMA models in self-supervised learning).

*`requires_grad=False` — per-tensor flag.* Defaults: `False` for new tensors, `True` for `nn.Parameter`. Rule: an op's output requires grad iff *any* input does.

```python
W = torch.randn(8, 4)                    # requires_grad=False
W_param = nn.Parameter(torch.randn(8,4)) # requires_grad=True
y = W @ x                                # requires_grad=False, no grad_fn
y2 = W_param @ x                          # requires_grad=True, has grad_fn
```

Use to freeze parameters: `for p in encoder.parameters(): p.requires_grad = False`.

*`detach()` — surgical cut.*

```python
y = encoder(x)                # y.requires_grad=True
y_frozen = y.detach()         # new tensor, same storage, requires_grad=False
loss = head(y_frozen).mean()
loss.backward()               # gradients flow into head, NOT into encoder
```

Same storage, no `grad_fn`, no requires_grad. Canonical **stop-gradient**:

- **Target networks in DQN/SAC** — target value `r + γ Q_target(s', a')` detached.
- **EMA encoders in BYOL/MoCo/DINO** — target encoder outputs detached.
- **Advantages in PPO** — detached so policy loss doesn't backprop through value computation.

`detach()` does **not** prevent in-place modifications from being observed: mutating a detached tensor in place is observed by autograd via the version counter. Treat detached views as read-only or use `.clone().detach()`.

*`torch.no_grad()` — region disable.*

```python
with torch.no_grad():
    val_loss = (model(val_x) - val_y).pow(2).mean()
```

Inside the block, every op behaves as if all inputs had `requires_grad=False`. Cheaper than evaluating with grad on, especially for memory.

*`torch.inference_mode()` — stricter, faster.* Drop-in for `no_grad()` for serving. Additionally:
- Skips **version counter** updates.
- Skips **view tracking**.

Tighter constraint: tensors created inside an `inference_mode()` block are **inference tensors** and cannot later be used in any autograd op. Try and you get an explicit error. Forces clean separation between training and serving paths.

Use `inference_mode` for pure inference. Use `no_grad`/`detach` when you compute eval loss inside training and want to mix results back.

*Performance ladder, fastest to slowest:*

1. `torch.inference_mode()`
2. `torch.no_grad()`
3. Tensors all having `requires_grad=False`
4. Default

*Comparison.*

| Tool                   | Scope      | Result usable in training graph? |
|------------------------|------------|----------------------------------|
| `requires_grad=False`  | Per-tensor | Yes                              |
| `detach()`             | Per-tensor | Yes (no grad through detach)     |
| `no_grad()`            | Code block | Yes                              |
| `inference_mode()`     | Code block | **No** (raises error)            |

*TraceML internal use.* TraceML's hooks shouldn't add tracked ops. When the layer-memory sampler computes a delta or clones a stat tensor, it should do so under `no_grad()` or via Python scalars / `.item()` so it doesn't accidentally extend the user's autograd graph.

**Concepts introduced:** requires_grad propagation rule, leaf freezing, `detach()` as stop-gradient with shared storage, in-place-modification footgun across detach, `no_grad()` as region disable, `inference_mode()` strictness (version counter + view tracking + inference-tensor type), inference tensor poison-pill semantics, performance ladder.

---

### P19: How does a tensor hook (`register_hook`) differ from a module hook (`register_forward_hook`)?

**Date:** 2026-04-24

**Short answer:** **Module hooks** fire around an `nn.Module`'s `__call__` — they see the module's *inputs and outputs as a whole* and run on the **main forward thread** (or autograd worker for backward hooks). **Tensor hooks** fire when the autograd engine produces a *gradient for that specific tensor* during backward — they run on an **autograd worker thread**, can mutate the gradient by returning a new value, and are the only way to intercept gradients flowing through a non-leaf tensor without forcing `.retain_grad()`. Module hooks are how TraceML attaches per-layer timing/memory; tensor hooks are how you'd intercept a specific gradient.

**Long answer:**

*Module hooks — bound to an nn.Module.*

- `register_forward_pre_hook(fn)` — `fn(module, inputs)` before forward.
- `register_forward_hook(fn)` — `fn(module, inputs, output)` after forward.
- `register_full_backward_pre_hook(fn)` — `fn(module, grad_output)` before backward.
- `register_full_backward_hook(fn)` — `fn(module, grad_input, grad_output)` after backward.

All four dispatch from `nn.Module.__call__` (`_call_impl`). Forward hooks fire on the main thread; backward hooks on autograd worker threads. Each hook sees whole-module input/output but **not** the parameter gradients of that module specifically — backward hook's `grad_input`/`grad_output` are gradients flowing through the module's input/output tensors.

This is why TraceML uses module hooks for *timing* and *activation memory* (properties of input/output flow) but reaches for parameter `.grad` directly (post-backward) for parameter gradient inspection.

*Tensor hooks — bound to a specific tensor's gradient.*

```python
y = some_layer(x)
def my_hook(grad):
    print("grad of y has norm", grad.norm())
    return grad.clamp(-1.0, 1.0)
handle = y.register_hook(my_hook)

loss = downstream(y).mean()
loss.backward()
handle.remove()
```

- Fires once per `.backward()`, when the engine produces a gradient for that tensor.
- If the function returns a new tensor, that replaces the gradient before propagating upstream.
- Runs on the autograd worker thread.
- Only way to inspect gradients of a non-leaf tensor without `.retain_grad()`.

Use cases: per-layer gradient-norm logging, custom gradient clipping at a specific point, gradient-reversal layers (return `-λ * grad`), debugging NaN gradients.

*Two flavors of tensor hook.*

- `tensor.register_hook(fn)` — fires when *that tensor's* gradient is produced.
- `tensor.register_post_accumulate_grad_hook(fn)` — fires after a leaf tensor's `.grad` has been accumulated into. PyTorch 2.x. Useful for "do something right after this parameter's gradient is final."

*Thread affinity.* Forward hooks: main thread. Backward and tensor hooks: autograd worker threads. If your hook touches a Python data structure also touched by main thread, you need thread-safety. TraceML's hook callbacks append to `Database` deques — `collections.deque.append` is documented thread-safe in CPython.

*Why TraceML uses module hooks for bulk instrumentation.* Module-level granularity matches the dashboard. Tensor hooks would mean N registrations per forward (every layer's output) with overhead, and the data already exists at module level. Tensor hooks reserved for surgical cases (specific gradient logging, custom interventions).

If TraceML adds per-parameter gradient statistics, the natural place is `register_post_accumulate_grad_hook` on each parameter — once per parameter, fires after gradient is final.

*Mental model.*

- **Module hook** = doorman at a building's entrance — sees everyone in/out, knows nothing about individual rooms (parameters).
- **Tensor hook** = tap on a specific water pipe — sees the gradient liquid flowing through that pipe, can change its flavor before passing it on.

Both are passive observers by default, both can intervene, both are how you instrument a dynamic graph framework — because the graph itself is too ephemeral to annotate.

**Concepts introduced:** module hook flavors and firing order, parameter-grad vs activation-grad distinction, tensor `register_hook` semantics, gradient mutation by hook return, `register_post_accumulate_grad_hook` for leaf tensors, autograd worker thread affinity, deque thread-safety as TraceML's design choice, gradient-reversal / NaN-detection / per-layer-norm canonical use cases.

---

## Optimizers

### P20: What state does an Adam-class optimizer hold per parameter, and why?

**Date:** 2026-04-24

**Short answer:** For each parameter tensor, Adam stores **two same-shape buffers** — `exp_avg` (first-moment estimate) and `exp_avg_sq` (second-moment estimate) — plus a scalar **`step` counter** for bias correction. AdamW adds nothing beyond that; variants like Adamax, NAdam, or Adafactor swap one of the buffers for a different statistic. The state lives in `optimizer.state`, a `defaultdict(dict)` keyed by **parameter tensor identity** (not name), and dominates optimizer memory (2× parameter size for Adam, before counting gradients).

**Long answer:**

*Where state lives.* `torch.optim.Optimizer` keeps `self.state: Dict[Tensor, Dict[str, Any]]` — keyed by parameter tensor *object* (Python `id` semantics). On the first `step()` for a given parameter, per-optimizer init code allocates the buffers and stashes them under that key. After step 1 of an Adam run on a 7B-parameter model, you have ~14B extra fp32 numbers — the "optimizer states" line item.

*Per-parameter dict for Adam.*

```python
state["step"] = torch.zeros((), dtype=torch.float32)  # or a Python int
state["exp_avg"] = torch.zeros_like(p, memory_format=...)        # m_t
state["exp_avg_sq"] = torch.zeros_like(p, memory_format=...)     # v_t
# Optional, only if amsgrad=True:
state["max_exp_avg_sq"] = torch.zeros_like(p, ...)
```

Implementation details: `step` is a 0-dim tensor by default so fused/foreach kernels can read it on-device without CPU sync. Buffers use `memory_format=torch.preserve_format` so channels-last conv weights keep their layout. `amsgrad` adds a third buffer (3× param size). Most people leave it off; "Adam = 2× params" is the rule of thumb.

*Why these buffers.* Adam's update is a low-pass filter over the gradient (`exp_avg`) divided by a low-pass filter over squared gradient (`sqrt(exp_avg_sq)`). Both are first-order IIR filters; computing the filter at step `t` requires the filter's previous output. The `step` counter exists for **bias correction** — at small `t` the EMA is biased toward zero (initialized at zero), so the update divides by `1 - beta^t` to debias.

*Variant cheat sheet.*

| Optimizer | Per-param state |
|---|---|
| SGD (no momentum) | none |
| SGD + momentum | `momentum_buffer` (1×) |
| Adam / AdamW | `exp_avg`, `exp_avg_sq`, `step` (2×) |
| Adam + amsgrad | + `max_exp_avg_sq` (3×) |
| Adafactor | factored row+col stats (~sqrt) |
| Lion | `exp_avg` only (1×) |

Adafactor's win: factored second moment. For a 4096×4096 weight, store row+col sums (8192 numbers) instead of a full-shape buffer (16M). That's why "Adafactor for memory-constrained training."

*Param identity is the gotcha.* `state` is keyed by tensor object. This breaks when:

- `model.to(device)` *after* the optimizer is constructed — when a parameter actually gets reallocated, `state` is keyed off a stale tensor and the new parameter starts fresh from zero state. Always: build model, move to device, *then* construct optimizer.
- Replacing a parameter mid-training — the old key sits in `state`, the new tensor has no entry.
- `optimizer.load_state_dict(...)` works because round-trips through *param_groups index ordering*, not tensor identity. `state_dict()` re-keys by integer position; `load_state_dict()` re-keys back using current param_groups. Parameter order between save and load must match.

*Memory accounting in practice.* For a 7B model in fp32 with AdamW:
- params: 28 GB
- gradients: 28 GB
- optimizer state: 56 GB
- **total before activations: 112 GB**

That 56 GB is exactly what ZeRO-1 / ZeRO-2 / FSDP shard. ZeRO-1 partitions just `optimizer.state` across ranks; ZeRO-2 also partitions gradients; ZeRO-3 partitions params. Reason ZeRO-1 alone is so effective: for Adam-class optimizers the per-param state is 2× the params — bigger than the gradients.

*TraceML angle.* When the "step" panel shows a sudden memory bump on the first optimizer step but flat thereafter, it's the lazy allocation in `_init_group` — buffers materialize on first use, not at construction. If you see optimizer-state allocation jump *every* step, somebody is reconstructing the optimizer in the loop, silently throwing away momentum.

**Concepts introduced:** `optimizer.state` defaultdict, per-parameter state dict, `exp_avg` / `exp_avg_sq` / `step` tensor, lazy state init in `_init_group`, parameter tensor identity as state key, bias correction, `amsgrad` extra buffer, `capturable` / CUDA-graph-friendly tensor step counter, fused/foreach optimizer kernels, optimizer state dict round-trip via param_groups index, ZeRO-1/2/3 sharding correspondence, Adafactor factored second moment, optimizer memory rule of thumb (2× params for Adam-class).

---

### P21: What does `optimizer.step()` actually do, and how does it know which gradients to use?

**Date:** 2026-04-24

**Short answer:** `optimizer.step()` iterates over the parameters it was given at construction time, **reads `p.grad` directly off each parameter tensor** (no name lookup, no graph traversal, no autograd involvement), and applies the update in place. The link between `loss.backward()` and `step()` is just the convention that backward writes into `param.grad`; `step()` then reads from there. There's no symbolic dependency — they're decoupled functions agreeing on the same attribute.

**Long answer:**

*The contract: backward writes `.grad`, step reads `.grad`.* This is the entire mechanism. Backward (see [P14](#p14-what-does-lossbackward-actually-do-step-by-step)) walks the autograd graph, accumulates `grad_output` into `param.grad` for every leaf with `requires_grad=True`. Optimizer reads from there. There's no shared registry, no observer, no callback — just two functions touching the same attribute slot.

This decoupling is why you can:
- Skip `step()` on some iterations (gradient accumulation, [P16](#p16-why-do-gradients-accumulate-by-default-when-is-that-useful-when-is-it-a-footgun)).
- Modify `.grad` between `backward()` and `step()` — clipping (`torch.nn.utils.clip_grad_norm_`), AMP scaling, masking, manual all-reduce.
- Call `step()` without `backward()` (e.g., manually populating `.grad` from ES or REINFORCE-style finite differences).
- Use different optimizers for different parameter subsets.

*The control flow inside `step()` (Adam, simplified).*

```python
@torch.no_grad
def step(self, closure=None):
    for group in self.param_groups:
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]
            if len(state) == 0:
                self._init_group(...)         # lazy alloc on first step
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])
        adam(params_with_grad, grads, exp_avgs, exp_avg_sqs, ...,
             beta1=beta1, beta2=beta2, lr=group["lr"], ...)
```

Three things to notice:

1. **`@torch.no_grad`** wraps the entire step. The weight update would otherwise create a new node in the autograd graph (`p_new = p - lr * grad`) and you'd backprop through your own optimizer.
2. **Group-by-group iteration** — outer loop is over `param_groups`, not all params. That's how per-group LR / weight decay / betas get applied (P22).
3. **Build flat lists, then call one function.** This is the **foreach pattern**.

*Three step kernels: for-loop, foreach, fused.*

- **For-loop (legacy).** Pure-Python loop. N CUDA dispatches per parameter → kernel-launch overhead per parameter. For a 200-layer model that's 200+ launches per step.
- **Foreach (`_foreach_*`).** `torch._foreach_add`, `_foreach_mul_`, etc. take *lists of tensors* and run as a single fused launch. ~1 launch per op type per group. Default since PyTorch 2.0 for CUDA tensors with matching dtypes.
- **Fused (`fused=True`).** The whole Adam update is one CUDA kernel. Available for Adam, AdamW, SGD on CUDA. Lowest overhead; locks you into that exact math.

If the optimizer step looks suspiciously cheap (sub-millisecond) on a giant model, `fused=True` is active.

*The closure parameter.* Some optimizers (LBFGS) need to re-evaluate loss multiple times per step for line search. `step(closure)` lets you pass a callable. For first-order optimizers, ignored.

*What `step()` does NOT do.*

- **Doesn't zero gradients.** Gradients persist after `step()`. Source of accumulation.
- **Doesn't synchronize with the GPU.** Queues kernels onto the current stream and returns. Host thread keeps moving.
- **Doesn't touch the autograd graph.** Graph is freed during `backward()`; by the time `step()` runs, no graph exists.
- **Doesn't do all-reduce.** In DDP, the all-reduce is hooked into `backward()` via gradient sync hooks. By `step()` time, gradients are already averaged.

*Why the design.* The `.grad`-as-the-interface contract feels almost too loose — no type-checking that backward and step agree, no enforcement of zero. But that's why PyTorch supports every weird gradient-modification trick (mixed precision, sparse updates, manual DDP, GAN alternating updates) without changing the optimizer interface. Looseness *is* extensibility.

*TraceML angle.* The step-time sampler measures wall-clock between optimizer-step boundaries. Spike on a particular iteration: (1) first iteration after fused-kernel JIT compile, (2) first iteration where lazy state allocation runs, (3) gradient clipping doing host-blocking norm, (4) sync point you didn't realize was there (logging `loss.item()`).

**Concepts introduced:** `.grad` as the backward↔step contract, `@torch.no_grad` wrapping, lazy state init, foreach (`_foreach_*`) vs fused vs for-loop kernels, kernel launch overhead per parameter, closure-based optimizers (LBFGS), decoupling of backward/clipping/accumulation/step, DDP gradient-sync hooks running inside backward (not step), why `step()` can be a sub-millisecond no-sync GPU dispatch.

---

### P22: What are `param_groups` for, and when do you reach for them (different LRs per layer, weight decay groups)?

**Date:** 2026-04-24

**Short answer:** `param_groups` is a list of dicts; each dict bundles a subset of parameters with its own hyperparameter overrides (`lr`, `weight_decay`, `betas`, `momentum`, etc.). The optimizer iterates over groups in `step()` and uses each group's hyperparameters for its parameters. You reach for them whenever a single global LR or weight-decay value is wrong: **discriminative learning rates** for transfer learning, **no-decay buckets** for biases / LayerNorm / embeddings, **LR multipliers** for newly added heads, **per-layer LR schedules** like LLRD.

**Long answer:**

*The data structure.*

```python
optimizer.param_groups = [
    {"params": [<Parameter>, ...], "lr": 0.001, "betas": (0.9, 0.999),
     "weight_decay": 0.01, "eps": 1e-8, ...},
    {"params": [<Parameter>, ...], "lr": 0.0001, "weight_decay": 0.0, ...},
]
```

When you do `Adam(model.parameters(), lr=1e-3)`, you get *one* group. When you do:

```python
Adam([
    {"params": backbone.parameters(), "lr": 1e-5},
    {"params": head.parameters(), "lr": 1e-3},
], weight_decay=0.01)
```

you get two groups. Per-group level wins; missing keys fall back to constructor defaults.

*How `step()` uses it.* From P21, the outer loop is `for group in self.param_groups:` and `lr`, `weight_decay`, `betas` are pulled from `group[...]` inside. There's no global LR — only group LRs. Mutating a group hyperparameter at runtime is picked up by the next `step()`:

```python
for g in optimizer.param_groups:
    g["lr"] *= 0.1
```

This is exactly what `torch.optim.lr_scheduler.*` does — schedulers store internal state and write into `param_groups[i]["lr"]` on each `scheduler.step()`. The optimizer has no concept of a schedule.

*Three workhorse use cases.*

**1. No-weight-decay buckets.** You do *not* want decay applied to biases, LayerNorm/BatchNorm gain and bias, or (debatably) embeddings. Standard split:

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
        no_decay.append(p)
    else:
        decay.append(p)
optim = AdamW([
    {"params": decay, "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
], lr=lr)
```

`p.ndim <= 1` heuristic catches biases and norm params (all 1-D). Every modern LM training recipe does this.

**2. Discriminative learning rates (transfer learning).** Pretrained backbone wants tiny updates, new head wants large updates:

```python
optim = AdamW([
    {"params": backbone.parameters(), "lr": 1e-5},
    {"params": classifier.parameters(), "lr": 1e-3},
])
```

Generalization: **layer-wise LR decay (LLRD)** for fine-tuning transformers — each layer gets `base_lr * decay_rate^(num_layers - layer_idx)`. One param_group per layer.

**3. Freezing without freezing.** Sometimes "almost freeze" — very low LR, no decay, conservative betas. Easier than toggling `requires_grad` mid-training (which interacts badly with parameter identity in optimizer state, P20).

*Other knobs.* For Adam: `lr`, `betas`, `eps`, `weight_decay`, `amsgrad`, `maximize`, `foreach`, `fused`, `capturable`, `differentiable`. For SGD: `lr`, `momentum`, `dampening`, `weight_decay`, `nesterov`. Anything in `defaults` can be per-group. You *cannot* mix optimizer classes — all groups in one optimizer use the same update rule.

*Schedulers and groups.* Most schedulers apply uniformly — multiply each group's LR by the same factor. Per-group schedules: construct one scheduler per group, or use `LambdaLR` with a *list* of lambdas:

```python
LambdaLR(optimizer, lr_lambda=[lam0, lam1, lam2])
```

The list form is canonical for "warm up backbone slowly while warming up head fast" in one scheduler call.

*Adding params after construction.* `optimizer.add_param_group({"params": ..., "lr": ...})` — useful for progressive unfreezing. New group inherits defaults.

*Common bugs.*

- **Same parameter in two groups.** PyTorch checks at construction and raises. Two *separate* optimizers over overlapping params: silent double-update.
- **Forgetting `weight_decay=0` for the no-decay group.** If only the decay group has it set and the constructor default is non-zero, both groups decay.
- **Manual `g["lr"]` then `scheduler.step()`.** Schedulers compute next LR from internal `last_lr`, not your override; your manual change gets overwritten.

*Why this design.* List-of-dicts is the cheapest possible generalization from "one global LR" to "per-subset LR" — no class hierarchy, no group manager, just iteration in `step()`. Composes cleanly with checkpointing, schedulers, gradient surgery. Loose, but composes — same philosophy as the `.grad` interface.

**Concepts introduced:** `param_groups` list-of-dicts, per-group hyperparameter overrides vs `optimizer.defaults`, no-decay bucket convention (biases / norms / 1-D params), `p.ndim <= 1` heuristic, discriminative learning rates, layer-wise LR decay (LLRD), `add_param_group`, schedulers writing into `group["lr"]`, `LambdaLR` with per-group lambdas, optimizer-class uniformity within one optimizer, double-update bug from overlapping groups.

---

### P23: Why prefer `zero_grad(set_to_none=True)` over the old default?

**Date:** 2026-04-24

**Short answer:** `set_to_none=True` deallocates the `.grad` tensors entirely (sets the attribute to `None`) instead of filling them with zeros in place. This skips a kernel launch per parameter, lets PyTorch reuse the freed memory for activations during the next forward, and turns "missing gradient" into a clean signal (`p.grad is None`) that the optimizer step can skip cleanly. As of PyTorch 2.0+, `set_to_none=True` is the **default**.

**Long answer:**

*What "zero" used to do.* Pre-2.0 default: `optimizer.zero_grad()` ran `p.grad.zero_()` for every parameter — in-place fill with zeros. For a 200-layer network: 200 separate CUDA kernel launches per step doing nothing but writing zeros. Each launch costs microseconds of host overhead. Pure overhead, no useful work.

*What `set_to_none=True` does.* Walks the parameters and runs `p.grad = None`. The previous gradient tensor's refcount drops to zero → CUDA caching allocator reclaims the memory → it's available for the next forward's activations. No kernel launch, no host-side dispatch loop, no zeros being written.

The next `loss.backward()` sees `p.grad is None` and **allocates a fresh gradient tensor**. So `zero_().` then `add_(grad)` becomes a single fused allocation that produces `grad` directly. Strictly less work.

*Three concrete benefits.*

1. **Eliminated per-parameter kernel launches.** N parameters → 0 zero-fill launches.
2. **Memory churn reduction.** Between `step()` and the next `backward()`, gradient tensors are dead weight. Releasing them lets the allocator hand that memory back to forward. For models near the memory ceiling, this is the difference between fitting and OOMing.
3. **Cleaner skip semantics in `step()`.** Look back at P21: `if p.grad is None: continue`. With `set_to_none=True`, parameters that didn't participate this iteration (frozen branch, conditional sub-network, MoE expert not routed to) have `p.grad is None` and the optimizer correctly skips — no spurious update. With `zero_()`, "didn't participate" looked identical to "participated and got zero gradient," and **AdamW would silently decay your frozen params**.

That third one is the subtle correctness benefit.

*Two gotchas.*

**Gradient accumulation.** Recall from [P16](#p16-why-do-gradients-accumulate-by-default-when-is-that-useful-when-is-it-a-footgun) that accumulation works *because* `backward()` adds into existing `.grad`. The pattern:

```python
for micro_batch in micro_batches:
    loss = model(micro_batch) / accum_steps
    loss.backward
optimizer.step
optimizer.zero_grad(set_to_none=True)  # only clear after all micro-batches
```

You must **not** call `zero_grad()` between micro-batches.

**Code that reads `p.grad` directly.** Custom code like `for p in model.parameters(): grads.append(p.grad.detach().clone())` blows up with `AttributeError: 'NoneType'` if any param didn't get a gradient this step. Defensive pattern:

```python
for p in model.parameters():
    if p.grad is None:
        continue
    grads.append(p.grad.detach().clone())
```

Same fix as inside the optimizer step. Anything walking `.grad` for monitoring (norm logging, custom clipping, TraceML's gradient sampler) should handle `None`.

*`optimizer.zero_grad()` vs `model.zero_grad()`.* Both exist, both accept `set_to_none`. Optimizer version walks `optimizer.param_groups`; model version walks `model.parameters()`. Equivalent if your optimizer covers all model parameters. Pick one consistently — most production code uses `optimizer.zero_grad()`.

*When to set `set_to_none=False`.* Almost never. Niche cases: downstream code requires `.grad` to be a tensor unconditionally, or you're capturing the optimizer step into a CUDA graph and want a fixed memory layout across iterations.

*Why this took a while to become default.* Backward-incompatibility risk. Lots of user code did `p.grad.zero_()` manually, or read `.grad` unconditionally, or relied on the gradient tensor being the same object across iterations. PyTorch flipped the default in 2.0 with the migration guide flagging the `is None` issue.

*TraceML angle.* When the step-memory sampler shows allocated memory dropping right after `zero_grad` and rising again on the next backward, that's `set_to_none=True` working. If memory stays flat across `zero_grad`, somebody passed `set_to_none=False` (or is on very old PyTorch) and is paying both per-param zero-fill cost and memory-occupied-during-forward cost.

**Concepts introduced:** `set_to_none=True` semantics, per-parameter kernel launch elimination, CUDA caching allocator memory reuse, `p.grad is None` as participation signal, AdamW weight-decay-on-zero-gradient bug avoidance, interaction with gradient accumulation, `AttributeError` migration footgun for code reading `.grad` unconditionally, `optimizer.zero_grad()` vs `model.zero_grad()` distinction.

---

## DataLoader

### P24: What's the relationship between `Dataset`, `Sampler`, and `DataLoader`?

**Date:** 2026-04-24

**Short answer:** **`Dataset`** answers "give me item *i*". **`Sampler`** decides *which* indices to ask for and in what order (sequential, random, weighted, distributed). **`DataLoader`** is the orchestrator that pulls indices from the sampler, calls the dataset for each, batches them via `collate_fn`, and optionally parallelizes the work across worker processes. Three orthogonal responsibilities — indexing, ordering, and batching/parallelism — composed into the iterator your training loop sees.

**Long answer:**

*The three roles, separated.*

- **`Dataset`** (`torch.utils.data.Dataset`) — pure index-to-sample function. Implement `__getitem__(idx) -> sample` and `__len__() -> int`. Content-addressable storage for samples.
- **`Sampler`** — iterator yielding indices. `SequentialSampler` yields `0, 1, 2, ...`. `RandomSampler` yields a permutation. `WeightedRandomSampler` samples per a weight vector. `DistributedSampler` yields a non-overlapping slice for the current rank (see [Q4](learning-qa.md#q4-what-is-a-gpu-rank)).
- **`BatchSampler`** wraps a `Sampler` and yields *lists* of indices.
- **`DataLoader`** — pull indices from the batch sampler → fetch each sample → run `collate_fn` → optionally pin and ship to GPU.

*Why this separation matters.* You can swap any of the three independently. Want deterministic eval order? `SequentialSampler`. Want class-balanced training? `WeightedRandomSampler`. Want multi-GPU sharding? `DistributedSampler`. The dataset code never changes. Strategy pattern applied to data loading.

*Map-style vs iterable-style.* PyTorch supports two flavors:

- **Map-style (`Dataset`)** — random access by index. Sampler decides indices. The common case.
- **Iterable-style (`IterableDataset`)** — yields samples one at a time, no `__getitem__`. Used for streams (Kafka, sharded record files where you only want sequential reads). Sampler is irrelevant; the dataset itself is the iterator. Multi-worker semantics differ — each worker reads its own shard; you handle the split inside the dataset.

*Concrete construction order.*

```python
loader = DataLoader(dataset, batch_size=32, sampler=RandomSampler(dataset),
                    num_workers=4, collate_fn=my_collate)

for batch in loader:        # 1. iter(loader) creates a _MultiProcessingDataLoaderIter
    train_step(batch)       # 2. spawns 4 worker processes
                            # 3. main process asks BatchSampler for 32 indices
                            # 4. main puts (task_id, indices) on a worker's input queue
                            # 5. worker calls dataset[i] for each, runs collate_fn
                            # 6. worker pushes batched result to shared output queue
                            # 7. main pulls results, optionally pins memory, yields
```

The sampler runs in the **main process**. Indices are cheap; the heavy lifting (`__getitem__`, decoding, `collate_fn`) runs in workers. Important — sampler state (RNG, epoch counter) lives in one place.

*RL analogy.* If you've used a replay buffer: `Dataset` is buffer storage (indexed cells), `Sampler` is the priority/uniform sampling policy, `DataLoader` is the actor's batched fetch + transfer. The buffer doesn't know about priorities; the prioritized sampler doesn't know about transition contents. Same factoring.

**Concepts introduced:** map-style vs iterable-style dataset, `Sampler`, `BatchSampler`, `DistributedSampler`, `WeightedRandomSampler`, strategy pattern for data loading, sampler runs in main process, indices vs samples flow through pipeline.

---

### P25: How does `num_workers > 0` actually work — threads, processes, IPC?

**Date:** 2026-04-24

**Short answer:** `num_workers=N` spawns **N child processes** (not threads — Python's GIL would defeat the purpose, see [Q6](learning-qa.md#q6-python-internals-bytecode-gil-cpu-bound-vs-io-bound)). The main process owns the sampler and one `multiprocessing.Queue` per worker for sending index batches; workers push completed mini-batches back through a single shared result queue. The start method (fork / spawn / forkserver, see [Q7](learning-qa.md#q7-spawning-fork-exec-and-multiprocessing-start-methods)) decides how those children come into existence — and on CUDA workloads it must not be plain `fork` after a CUDA context exists in the parent (see [Q11](learning-qa.md#q11-what-is-a-cuda-context-and-why-is-it-fork-unsafe)).

**Long answer:**

*Why processes, not threads.* Data loading is mostly Python — PIL decodes, NumPy augments, tokenizers regex. All CPU-bound Python = serialized by the GIL. Threads would give near-zero parallelism. Subprocesses each have their own interpreter and own GIL → real parallelism, at the cost of needing IPC.

*The startup dance.* When you do `for batch in loader:` and `num_workers > 0`, PyTorch instantiates `_MultiProcessingDataLoaderIter`:

1. Creates `N` `multiprocessing.Queue` objects — one **index queue** per worker (main → worker, carrying `(task_id, list_of_indices)`).
2. Creates one shared **data queue** (worker → main, carrying `(task_id, batched_sample)`).
3. Calls `multiprocessing.Process(target=_worker_loop, ...)` N times.
4. Pre-queues a few index batches per worker (`prefetch_factor`, default 2).

Each worker:

```python
while True:
    task = index_queue.get()             # block waiting for indices
    if task is None: break               # poison pill = shutdown
    task_id, indices = task
    samples = [dataset[i] for i in indices]
    batch = collate_fn(samples)
    data_queue.put((task_id, batch))     # push back to main
```

Main process pulls from `data_queue`, reorders by `task_id` so batches return in sampler order, yields the batch. Pushes the next index batch onto the freed worker.

*The IPC payload — how a tensor crosses process boundaries.* A `torch.Tensor` crossing a `multiprocessing.Queue` doesn't get serialized byte-for-byte. PyTorch installs custom reducers that:

1. Allocate a **shared-memory** region (POSIX `shm_open` on Linux/Mac) and copy the tensor's storage there.
2. Serialize only the metadata: shape, dtype, stride, the shared-memory region's name/fd.
3. Receiving process reconstructs a `Tensor` whose storage points at the same shared-memory region — no copy.

Moving a 4 MB image batch is roughly: one `memcpy` into shared memory in the worker, one fd handoff over the queue, zero copy on receive. The fd handoff uses `SCM_RIGHTS` over a Unix domain socket on Linux.

*Start methods.*

- Linux: historically `fork`. PyTorch 2.x still uses fork unless you change it. Fast (COW) but unsafe with CUDA.
- macOS: `spawn` since Python 3.8.
- Windows: `spawn` only.

If `torch.cuda.init()` (or anything triggering context creation — `model.cuda()`, `tensor.to('cuda')`, even `torch.cuda.is_available()` in some versions) ran before the iterator is created, fork is dangerous. Workers inherit corrupt CUDA state. Fixes: (a) don't touch CUDA in workers; (b) `multiprocessing.set_start_method("spawn")` before DataLoader construction; (c) `forkserver`.

*Worker init.* `worker_init_fn` runs in each worker after spawn — typically used to set per-worker RNG seeds. Otherwise all workers produce identical augmentations because they inherited the same RNG state on fork. `torch.utils.data.get_worker_info()` exposes the worker's id and seed.

*RL analogy.* `num_workers` is the actor pool in IMPALA / Ape-X. Main = learner; workers = actors generating data; queues = trajectory pipes. Same questions: how many actors, how to bound the queue (`prefetch_factor * num_workers`), how to seed RNG so actors don't produce correlated trajectories. `worker_init_fn` seed-derivation is the "give each actor distinct RNG" pattern.

*Cost.* Worker startup is *not* free. Each forks/execs, imports torch, imports your dataset module, opens dataset files. By default, this happens **every epoch** — workers torn down at end of `for batch in loader:` and re-spawned next. That's what `persistent_workers` solves (P28).

**Concepts introduced:** `_MultiProcessingDataLoaderIter`, index queue vs data queue, `prefetch_factor`, poison-pill shutdown, shared-memory tensor passing, custom reducers, `SCM_RIGHTS` fd passing, `worker_init_fn`, per-worker RNG derivation, fork-vs-spawn implications for DataLoader workers, actor pool analogy.

---

### P26: What is `pin_memory=True` and when does it help?

**Date:** 2026-04-24

**Short answer:** `pin_memory=True` tells the DataLoader to allocate the returned batch's CPU storage in **page-locked (pinned) memory** instead of normal pageable RAM. Pinned memory has a fixed physical address the OS can't swap out, which lets the CUDA driver DMA-copy it directly to the GPU **asynchronously on a non-default CUDA stream** — overlapping the H2D transfer with GPU compute on the prior batch (see [Q15](learning-qa.md#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread)). It only helps when you actually do `batch.to(device, non_blocking=True)` and have GPU work to overlap with.

**Long answer:**

*Why "pinning" exists.* Your process's memory is virtual. The OS can move physical pages around, swap them, COW them. When the GPU does a DMA copy from CPU RAM, it talks to a physical address — no concept of your page table. So if asked to copy a normal pageable buffer:

1. **Slow path:** lock buffer page-by-page, wait for OS to give a stable physical address, copy synchronously, unlock. CPU thread blocks.
2. **Fast path:** copy first into a driver-internal pinned bounce buffer, then DMA from there. Two copies, but the second can be async.

If you allocate the buffer pinned **upfront**, neither bounce nor block is needed: driver DMAs straight from your buffer, on a copy stream, while compute runs on another stream. That's the win.

*What `pin_memory=True` actually does.* After a worker hands back a batch, the main process iterates the batch (recursing into dicts/lists/tensors) and calls `tensor.pin_memory()` on each. Allocates a new pinned region (`cudaHostAlloc`) and copies the tensor in. This *adds* a CPU-side copy (worker shared-memory buffer → pinned buffer). The bet: the small synchronous copy on CPU is paid back by enabling async H2D copy that overlaps with GPU compute.

There's also a dedicated **pin_memory thread** in the main process doing this work in the background, so the next batch can be pinned while your training step runs.

*Why it requires `non_blocking=True`.*

```python
for batch in loader:                          # batch.x is pinned CPU tensor
    x = batch.x.to('cuda', non_blocking=True) # DMA queued, returns immediately
    y = batch.y.to('cuda', non_blocking=True)
    out = model(x)                            # queued; CUDA inserts an event-wait
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
```

- `to('cuda', non_blocking=True)` on a pinned tensor: CUDA queues the H2D copy and returns. CPU does not wait.
- `model(x)` queues compute. The kernel won't run until `x` is on-device, but the *queueing* is non-blocking.
- While GPU computes batch N, the next iteration starts: the loader hands back batch N+1, the pin thread had it pinned already, the H2D copy starts on the copy stream and runs in parallel.

This overlap is the entire point.

If you pass `non_blocking=False` (the default!), the copy is synchronous from the CPU's perspective — `to()` doesn't return until GPU has the data. Pinning still avoids the bounce-buffer copy, but you've thrown away the overlap. Most of the win is gone.

*When pinning hurts or doesn't help.*

- **No GPU at all** — wasted work.
- **Tiny tensors** — overlap window too small.
- **Memory-constrained host** — pinned memory is pinned (can't be swapped, can't be reclaimed). System-wide limit (`ulimit -l`). Pinning huge dataloader buffers can OOM the system.
- **Already on GPU** — pin_memory moot.

*TraceML cross-check.* If H2D is the bottleneck, the symptom is GPU utilization sags between steps and TraceML's step-time decomposition shows a "data wait" gap. P29 covers diagnosis.

*Mental model.* Without pinning: you mail the GPU a package, but it's in your pocket and you have to walk it over. With pinning + non_blocking: you dropped the package in a designated outbox; the courier (DMA engine) picks it up and delivers on its own stream. The trick: you must actually leave the outbox — `non_blocking=True` is what tells you not to stand there waiting.

**Concepts introduced:** page-locked (pinned) memory, DMA, bounce buffer, `cudaHostAlloc`, `pin_memory` thread in DataLoader, `non_blocking=True`, async H2D copy on a non-default stream, host/device transfer overlap, ulimit on locked memory, when pinning hurts.

---

### P27: What's `collate_fn`, and when do you need a custom one?

**Date:** 2026-04-24

**Short answer:** `collate_fn` is the function the DataLoader calls to turn a `list[sample]` into a single batched object — typically by stacking tensors along a new batch dimension. The default (`default_collate`) handles tensors, numpy arrays, scalars, dicts, and lists recursively, assuming every sample has the **same shape**. You write a custom `collate_fn` whenever (a) samples have variable shape and need padding, (b) you want non-tensor metadata to come through unchanged, or (c) you want to do batch-level work cheaply in worker processes.

**Long answer:**

*Where collate sits.* `dataset[i]` returns one sample. The worker gathers `B` samples into a Python list, then calls `collate_fn(list_of_samples)`. The result is what your training loop sees as `batch`. **Critically, `collate_fn` runs in the worker process** — heavy collate work parallelizes with `num_workers`.

*What `default_collate` does.* Recurses by type:

- `torch.Tensor` samples → `torch.stack(samples, dim=0)`. Requires identical shapes.
- numpy array → convert to tensor, then stack.
- Python scalars → `torch.tensor([...])`.
- `Mapping` (dict) → `{k: default_collate([s[k] for s in samples]) for k in samples[0]}`.
- `Sequence` (tuple, list) → `tuple(default_collate(field_i_across_samples) for each i)`.
- Custom objects → `TypeError`.

Sample `(image: Tensor[3,224,224], label: int)` → `(Tensor[B,3,224,224], Tensor[B])`. Sample `{"input_ids": Tensor[L], "labels": int}` and **all samples have same `L`** → `{"input_ids": Tensor[B,L], "labels": Tensor[B]}`.

*When the default breaks.*

**Variable-length sequences.** NLP samples have different `L`. `torch.stack` fails. You need to pad to max length in batch and probably build an attention mask:

```python
def collate_pad(samples):
    lens = [s["input_ids"].shape[0] for s in samples]
    max_len = max(lens)
    pad = torch.zeros(len(samples), max_len, dtype=torch.long)
    mask = torch.zeros(len(samples), max_len, dtype=torch.bool)
    for i, s in enumerate(samples):
        pad[i, :lens[i]] = s["input_ids"]
        mask[i, :lens[i]] = True
    return {"input_ids": pad, "attention_mask": mask,
            "labels": torch.tensor([s["labels"] for s in samples])}
```

**Variable-size images** (detection, segmentation). Keep as a list, model handles it (Mask R-CNN style). Default collate would refuse.

**Graph data.** Need to concatenate node tensors and offset edge indices. PyTorch Geometric ships `Batch.from_data_list`.

**Non-tensor metadata** (filenames, image IDs for evaluation). Default collate errors or wraps unhelpfully. Custom collate keeps them as Python lists.

**Batch-level preprocessing in workers.** Sorting by sequence length (for packed RNN), bucketing, batch-wide augmentations. Doing it in `collate_fn` runs in the worker, in parallel.

*The async H2D angle.* Because collate runs in workers, resulting tensors live in the worker's address space. They cross to main via shared memory (P25). The pin_memory thread (P26) re-pins them. None of this requires you to think about it — but it's why `collate_fn` should output **tensors that are contiguous in memory and on CPU**. Non-contiguous or GPU tensors break the pin path.

*Pitfalls.*

- **Returning Python lists of tensors instead of stacked tensors** — works, but defeats vectorization downstream and skips the pin-memory fast path.
- **Building tensors with `torch.zeros(...).pin_memory()` inside collate** — wrong. Pinning happens in the main process *after* collate.
- **Calling tokenizers per-sample when you could batch them** — HF tokenizers have a fast batch mode. Better: collect strings, call `tokenizer(texts, padding=True)` once.
- **Heavy CPU work in collate, low `num_workers`** — increase workers or move work into `__getitem__`.

*RL analogy.* If `__getitem__` is env's `step()` returning a transition, `collate_fn` is the replay buffer's batch-sampling logic stacking transitions into `(s, a, r, s')` tensors. Per-item logic vs batch-shape logic.

**Concepts introduced:** `default_collate` recursion rules, padding/masking for variable-length batches, collate-runs-in-worker, batch-level preprocessing in workers, contiguous-CPU-tensor requirement for the pin path, common collate pitfalls.

---

### P28: What is `persistent_workers`, and what problem does it solve?

**Date:** 2026-04-24

**Short answer:** `persistent_workers=True` keeps the DataLoader's worker processes alive between epochs instead of tearing them down at the end of `for batch in loader:` and re-spawning. The problem this solves is worker-startup cost: forking/execing N processes, importing torch and your dataset module in each, opening dataset files, warming filesystem caches. Big wins on small-to-medium datasets where epoch length is comparable to worker startup time, or when `__init__` of your dataset is expensive.

**Long answer:**

*The default lifecycle.* Without `persistent_workers`, the iterator returned by `iter(loader)` owns the workers. When the iterator is GC'd (end of `for` loop), `__del__` sends a poison-pill `None` down each index queue, joins workers, closes queues. Next epoch, `iter(loader)` again triggers the whole spawn dance from P25.

This is wasteful when:

- Dataset's `__init__` is expensive (loads a manifest from S3, builds an index, opens a multi-GB memory-mapped file).
- First few `__getitem__` calls are slow because OS page cache is cold.
- Many epochs of a small dataset.

A typical 8-worker setup on a fat dataset can spend 5–15 seconds per epoch on worker startup alone. ×100 epochs = 10+ minutes burned on nothing.

*With `persistent_workers=True`.* Workers reused across epochs. Index queues intact. Dataset state in each worker preserved. PyTorch tells each worker "new epoch starts now" via a sentinel so it can reset per-epoch state — most importantly, the `Sampler` is re-iterated in main process and new index batches are pushed.

*Subtleties.*

- **Sampler interaction.** `DistributedSampler` needs `set_epoch(epoch)` called on it manually for shuffle to actually change between epochs. People sometimes assume re-spawning would reset something.
- **State leaks.** If your dataset opens file handles in `__init__` and you rely on them being re-opened each epoch (to pick up new files in a directory), persistent workers won't see new files.
- **Memory growth.** A leaky dataset (caches that grow per `__getitem__`) leaks across epochs instead of being reclaimed by worker death.
- **CUDA contexts in workers.** If a worker ever initialized CUDA (rare, e.g., GPU-side decoders), keeping it persistent means that context lives forever. Usually fine.

*Interaction with `prefetch_factor`.* `prefetch_factor` controls how many batches each worker has queued ahead. Together with persistent workers, you get a steady-state pipeline: workers always have N batches in flight; at epoch boundaries the pipeline flushes once and refills.

*When to leave it `False`.* Default is `False` and that's a deliberate conservative choice:

- One-shot evaluation runs (no second epoch).
- Datasets that genuinely change between epochs (active learning).
- Memory-tight setups where worker process growth across many epochs would OOM.
- Debugging.

For long training runs on a stable dataset, `persistent_workers=True, num_workers=N, prefetch_factor=2` is the canonical fast-path config.

*RL analogy.* The difference between **resetting the actor pool every iteration** vs **keeping persistent actors that you just nudge** between learner updates. Long-lived actors keep replay caches warm, env state warm, avoid env reconstruction. The cost: "new iteration" semantics are explicit, not implicit-via-restart.

**Concepts introduced:** worker process lifecycle, per-epoch teardown/respawn cost, page-cache cold-start cost, `persistent_workers` semantics, `set_epoch` on `DistributedSampler`, latent-leak exposure with persistent workers, steady-state pipeline with `prefetch_factor`.

---

### P29: Why does DataLoader sometimes stall a training step, and what does TraceML look at to diagnose this?

**Date:** 2026-04-24

**Short answer:** A "DataLoader stall" means your training step's wall-clock time is dominated by **waiting for the next batch from the loader**, not by forward/backward/optimizer or all-reduce. Classic signature: GPU utilization drops to near-zero between steps, host time inside `next(iterator)` is large, workers' result queue is empty when main asks. TraceML diagnoses this by decomposing step time into phases — **dataloader-iter time, forward, backward, optimizer, sync** — and surfacing the dataloader phase as a distinct line, plus per-step GPU idle gaps.

**Long answer:**

*Why a stall happens.* DataLoader pipeline is producer-consumer. Workers produce; main consumes. Steady-state happiness requires producers' throughput ≥ consumer's, with `prefetch_factor` worth of slack to absorb jitter.

A stall: **consumer asks for a batch, producer queue is empty, consumer blocks**. The blocking happens inside `next(iterator)` or implicit `__next__` of `for batch in loader:`. From the GPU's perspective: kernel queue drains, no new work, GPU goes idle.

Producer-slow causes:

- **CPU-bound `__getitem__`** — image decoding, tokenization. Workers at 100% CPU, too few of them. Fix: more workers, simpler augmentations, GPU-side ops.
- **I/O-bound `__getitem__`** — slow disk, S3, NFS. Workers in `D` (uninterruptible sleep) state. CPU low but throughput low. Fix: more prefetch, faster store, local caching.
- **Insufficient `prefetch_factor`** — bursty workers with no buffer. Fix: bump from 2 to 4 or 8.
- **Worker startup cost** — first batches of an epoch are slow because workers just spawned. Fix: `persistent_workers=True` (P28).
- **GIL contention in main** — model code or logging saturates the CPU; can't drain result queue. Fix: move work off the main thread.
- **H2D not overlapped** — pin_memory off, or `non_blocking=False` (P26).
- **Disk thrash** — workers' reads evict each other from cache.

*The diagnostic decomposition.*

```
step_time = dataloader_wait        # next(iterator) blocking
          + h2d_transfer           # batch.to('cuda') (visible if non_blocking=False)
          + forward                # model(x)
          + backward               # loss.backward()
          + ddp_sync               # all-reduce wait (see Q12)
          + optimizer              # opt.step()
          + overhead               # logging, metric updates
```

The art: figure out which term is large and chase only that.

*What TraceML measures.*

- **Step boundaries.** `trace_step()` from [traceml/src/traceml/decorators.py](https://github.com/Pendu/traceml/blob/main/src/traceml/decorators.py) marks each step. The step-time sampler records wall-clock per step.
- **Dataloader iteration time.** TraceML monkey-patches `DataLoader.__iter__` (and the iterator's `__next__`). Each `next()` call timed. Time **between previous step's end and the next batch arriving** is the dataloader-wait term. Smoking gun for "dataloader-bound."
- **Forward / backward / optimizer.** Layer-level forward and backward hooks ([Q9](learning-qa.md#q9-what-are-hooks-and-what-does-injecting-hooks-in-process-mean)) record CUDA events at module boundaries — host time and GPU time both measured (see [Q15](learning-qa.md#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread)). Optimizer step patched similarly.
- **GPU idle gap.** Derived from "wall-clock − summed GPU time." Growing GPU idle gap with stable forward/backward GPU time = dataloader-stall signature.
- **System sampler.** Samples CPU and GPU utilization at fixed interval. Low GPU util + high worker-process CPU corroborates "workers can't keep up."

*Disambiguating dataloader-stall from look-alikes.*

| Symptom | Likely cause | TraceML signal |
| --- | --- | --- |
| Long `next(iter)` time, GPU idle, low GPU util | Dataloader-bound | Large dataloader-iter, stable fwd/bwd |
| Long `optimizer.step()` host time | CPU-bound optimizer | Optimizer phase grows |
| Long backward, GPU idle near end | All-reduce wait (DDP) | DDP sync phase grows; see [Q12](learning-qa.md#q12-what-is-nccl-all-reduce-and-why-is-it-a-barrier) |
| Long forward host time, short GPU time | Python dispatch overhead | host >> GPU per layer |
| First step of epoch slow only | Worker startup | Stall epoch-start, gone after; fix with `persistent_workers` |

*Concrete recipe.*

1. Total step time vs. forward+backward+optimizer GPU time. If total >> GPU time, suspect dataloader-bound or sync-bound.
2. Look at dataloader-iter time. If dominant → dataloader-bound. Stop, fix loader.
3. Else, look at GPU vs host time per layer. Host >> GPU = framework dispatch overhead (consider compile/scripting). GPU >> host = compute-bound.
4. If forward/backward/optimizer all short but step time large → DDP sync time, all-reduce blocking.

*Why this matters for product.* "Where is my training time going?" is the question every PyTorch user asks. TraceML's competitive advantage: this question gets answered automatically — no manual `time.perf_counter()` peppering, no PyTorch profiler trace to load into Chrome. The phase decomposition *is* the product.

*Mental model.* The training loop is an assembly line. Dataloader is the parts conveyor; GPU is the worker. If parts arrive too slowly, the worker stands idle — even though the worker is fast. Fix is never "make the worker faster"; it's "fix the conveyor." TraceML's job: point a finger unambiguously at the conveyor.

**Concepts introduced:** producer-consumer model, dataloader stall signature, step-time phase decomposition, dataloader-iter timing via `__iter__`/`__next__` patching, GPU idle gap as derived metric, disambiguation table, "fix the conveyor not the worker" mental model.

---

## CUDA dispatch path & memory model

### P30: How does `y = torch.relu(x)` get from Python all the way to a CUDA kernel (the dispatcher path)?

**Date:** 2026-04-24

**Short answer:** Python's `torch.relu(x)` resolves to a **C-extension function** in `_C`, which wraps the tensor as a **THPVariable** and calls into the **C++ ATen dispatcher**. The dispatcher consults a per-operator **dispatch table** keyed by **DispatchKey** (Autograd, CUDA, CPU, etc.), walks the keys in priority order, calls the **autograd kernel** (which records the op in the graph), then re-dispatches and lands on the **CUDA kernel registration**, which finally **enqueues a kernel launch** on the current CUDA stream and returns to Python — usually before the GPU has touched a single byte.

**Long answer:**

*From Python to C: the THPVariable boundary.* Python objects representing tensors are **THPVariable** instances — `PyObject` subclasses that hold an `at::Tensor` (the C++ tensor) inside. `torch.relu(x)` descends into a C function that unpacks the THPVariable, pulls out the `at::Tensor`, and calls `at::relu(x)`. Step one: leave the Python interpreter, enter PyTorch's C++ world.

*ATen — the tensor library.* Each operator is declared once in YAML schema (`native_functions.yaml`) and generated into a uniform C++ API. `at::relu(x)` is **not** a direct call to a CPU or CUDA implementation. It is a **dispatch stub**: it asks the dispatcher "given this tensor, which kernel should actually run?"

*The dispatcher and DispatchKey.* The C++ dispatcher is a per-process registry mapping `(operator, DispatchKey) -> kernel function pointer`. A **DispatchKey** is an enum entry: `CPU`, `CUDA`, `Autograd`, `AutogradCUDA`, `XLA`, `Functionalize`, `AutocastCUDA`, etc. Every tensor carries a **DispatchKeySet** — a 64-bit bitset of which keys apply. `requires_grad=True` adds `Autograd`; being on CUDA adds `CUDA` and `AutogradCUDA`; being inside `torch.autocast` adds an autocast key.

When `at::relu(x)` runs, the dispatcher computes the union of dispatch keys across inputs, **masks out keys excluded by local dispatch state** (`torch.no_grad()` excludes Autograd), and walks the remaining set in **priority order**. Each kernel can re-dispatch by clearing its own key. No `if cuda: ... else: ...` ladder anywhere.

*The actual walk for `relu(x)` on a CUDA tensor with grad.*

1. **`AutogradCUDA` kernel runs first.** Saves what it needs for backward, creates a `ReluBackward` node in the autograd graph, records it as `y.grad_fn`, and **redispatches** with the Autograd key removed.
2. **`CUDA` kernel runs.** The "real" implementation, registered for relu under DispatchKey::CUDA. Calls `at::native::relu_cuda(x)`.
3. The CUDA implementation **launches a kernel** — `relu_kernel<<<grid, block, 0, stream>>>(x.data_ptr(), y.data_ptr(), n)`. Behind the syntax is `cudaLaunchKernel`, which **enqueues** the launch on the current CUDA stream and returns immediately.
4. C++ stack unwinds. The new THPVariable is constructed from the `at::Tensor`. Python receives `y`.

At the moment Python receives `y`, the GPU has likely **not** finished — possibly not started — the relu computation. `y` is a handle to memory the kernel will fill, with stream-side ordering guarantee.

*Kernel registration.* Kernels register at process startup via `Dispatcher::registerImpl(operator, key, fn)`. `TORCH_LIBRARY_IMPL(aten, CUDA, m) { m.impl("relu", relu_cuda); }` — runtime registration into a hash table. Backends like XLA, MPS, third-party accelerators register the same way.

*Why this matters for instrumentation.*

- **`nn.Module` forward hooks** sit at the Python `__call__` level — way above the dispatcher. They see "the layer ran" but not individual ops, and they fire even for ops the user wrote outside any module.
- **Functional ops called outside an `nn.Module`** never trip a module hook. They go straight through the dispatcher.
- To observe at the op level you plug into the dispatcher itself. PyTorch exposes this via **`__torch_dispatch__`** and via the **profiler** (which registers a special dispatch key).
- TraceML operates at the module/patch level (cheap, broad), not the dispatcher level. Knowing the dispatcher path tells you whether a future "per-op" mode is worth the overhead.

*Mental model.* The dispatcher is a tiny **virtual function table per operator**, except the dispatch axis is "DispatchKeySet" not "object type." Autograd, autocast, vmap, functionalization are all just keys layered on top, each kernel doing its thing then re-dispatching. The CUDA "kernel" at the bottom is the last entry the dispatcher reaches.

**Concepts introduced:** THPVariable, `torch._C` extension, ATen, `at::Tensor`, `native_functions.yaml`, C++ dispatcher, DispatchKey, DispatchKeySet, dispatch priority order, kernel registration (`TORCH_LIBRARY_IMPL`), autograd kernel as a dispatch key, redispatch, kernel launch (`cudaLaunchKernel`), `__torch_dispatch__`, dispatcher-level vs module-level instrumentation.

---

### P31: What is ATen, and what is the C++ dispatcher? Why does it matter for instrumentation?

**Date:** 2026-04-24

**Short answer:** **ATen** is PyTorch's C++ tensor library — the canonical implementation of every tensor operator, declared once in a schema and generated into a uniform C++ API. The **C++ dispatcher** is the small runtime registry mapping `(operator, DispatchKey) -> kernel function pointer` and decides, for each call, which backend kernel actually runs. For instrumentation it draws a sharp line: anything you intercept above the dispatcher is cheap but coarse; anything below it requires hooking the dispatcher, which is more powerful but more fragile against PyTorch internals.

**Long answer:**

*ATen.* Lives under `aten/src/ATen/`. Two things bundled: (a) schema + generated API surface from `native_functions.yaml`, and (b) the directory of backend implementations (`aten/src/ATen/native/cuda/`, `aten/src/ATen/native/cpu/`). `at::Tensor` is the single C++ tensor type; every backend produces and consumes it.

*The dispatcher.* Conceptually one big hash table:

```
(OperatorHandle, DispatchKey) -> KernelFunction
```

Every operator gets an `OperatorHandle`. Every backend registration adds an entry. At call time:

1. Compute **DispatchKeySet** from the inputs (and thread-local state — autograd? autocast? `no_grad`? functorch transforms?).
2. Pick the **highest-priority key** in that set, look up `(op, key)`.
3. Call that kernel. Re-dispatch by clearing the key.

Priority (high to low, sketch): **functorch transforms** → **Python (`__torch_dispatch__`)** → **Functionalize** → **Autocast** → **Autograd** → **Backend (CUDA/CPU)**. Each layer is a pass over the op: vmap turns scalar ops into batched ones, autocast inserts dtype casts, autograd records graph nodes, backend does the math. They compose by re-dispatching, never by knowing about each other.

*Why the dispatcher exists.* Solves a real problem: ~2,000 operators × N backends × M cross-cutting features (autograd, autocast, vmap, functionalization, profiling). Without it you'd have combinatorial explosion of `if cuda and autocast and grad and vmap: ...` in every op. Each operator has one schema; each backend registers kernels for the ops it implements; each cross-cutting feature is a key with its own kernels; they compose by re-dispatching. Chain-of-responsibility pattern with priority-ordered handlers.

*Why this matters for instrumentation — the layered view.*

| Layer                                  | Granularity                  | Cost   | Coupling to PyTorch internals |
| -------------------------------------- | ---------------------------- | ------ | ----------------------------- |
| Training loop wrappers (HF/Lightning)  | Steps                        | ~zero  | Low                           |
| `nn.Module` hooks                      | Per layer                    | Low    | Low (stable public API)       |
| Monkey-patches on `DataLoader`/optim   | Per call                     | Low    | Medium (private internals)    |
| C++ dispatcher hooks (profiler key)    | Per op                       | Medium | High                          |
| `__torch_dispatch__` in Python         | Per op, in pure Python       | High   | High                          |
| CUPTI / CUDA event timing              | Per kernel launch on GPU     | Medium | High                          |

PyTorch's profiler plugs in at the dispatcher level — registers a "profiler" handler that records start/end of every op. That's why it sees `aten::add` calls but also why it has measurable overhead.

TraceML lives in the top three rows: module hooks for layer visibility, monkey-patches for dataloader/optimizer/forward boundaries, training-loop integrations. Stays *above* the dispatcher. Wins: cheap, robust across PyTorch versions. Costs: invisible to ad-hoc functional calls (`y = torch.matmul(a, b)` outside any module — see P30) and unable to attribute time to specific ATen ops.

*Why the dispatcher is fragile from outside.* *Technically* public (third-party backends use it) but C++ symbols, key enum, and registration macros are unstable across releases. The "PyTorch coupling" risk in the project constraints — every PyTorch version requires regression check. Module hooks and Python-level patches are far more stable.

*What `__torch_dispatch__` gives you.* Subclass `torch.Tensor`, define `__torch_dispatch__(cls, func, types, args, kwargs)`, your method gets called on every op invoked on instances. How `torch.fx`, `torch.compile`'s AOTAutograd, ProxyTensor implement themselves. *The* extension point for op-level Python instrumentation.

*Mental model.* The dispatcher is the thinnest possible "trait dispatch" mechanism for tensors, with the trait being "what is your backend + what cross-cutting features apply." ATen is the giant menu of operations. Together they let PyTorch present **one** Python API while supporting CPU, CUDA, MPS, XLA, ROCm, autograd, autocast, vmap, functionalization, tracing — without any of those layers needing to know about the others. The instrumentation question reduces to "at which layer of the onion do I want to listen?"

**Concepts introduced:** ATen as schema + generated API + native kernels, `native_functions.yaml`, C++ dispatcher as `(op, key) -> kernel` table, DispatchKeySet from inputs + thread-local state, dispatch priority order, re-dispatch mechanism, backend-as-keyset, chain-of-responsibility pattern, `__torch_dispatch__` Python extension point, profiler-as-dispatch-key, instrumentation layer trade-off table.

---

### P32: What does `torch.cuda.synchronize()` actually do, and why is it expensive?

**Date:** 2026-04-24

**Short answer:** `torch.cuda.synchronize()` blocks the calling **CPU thread** until **all previously queued work on all CUDA streams of the current device** has finished executing on the GPU. It is expensive because it forces a host↔device **synchronization barrier** that drains the asynchronous pipeline — destroying the CPU/GPU overlap that is the whole point of the [stream model](learning-qa.md#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread). Necessary for correct timing measurements, but should never appear in a hot loop in production.

**Long answer:**

*The async pipeline you're breaking.* A CUDA op is a launch queued on a stream; returns to Python before the GPU runs it. Under steady state, the CPU is hundreds of ops ahead of the GPU. This **pipelining** hides Python and dispatcher overhead behind GPU compute. As long as the queue has work, host time is "free."

`torch.cuda.synchronize()` says: "Stop. Wait until the queue is empty and all in-flight kernels on every stream are done." Mechanically: `cudaDeviceSynchronize()` under the hood — a CUDA driver call that blocks the host thread until the device reports completion.

*Why it's expensive — three costs.*

1. **You lose pipelining.** During the sync the CPU is idle (waiting); after the sync the GPU is idle (no queued work) until the CPU dispatches more. If you sync every iteration: total time becomes `host_time + gpu_time` instead of `max(host_time, gpu_time)`. For a step where Python takes 10 ms and GPU takes 30 ms, async gives you ~30 ms; per-step sync gives you ~40 ms.
2. **The GPU drains and refills.** Modern GPUs are happiest with deep work queues. After synchronize the queue is empty. The next kernel pays full launch latency (~5-20 µs) before any SM starts. Death by a thousand kernel-launch overheads.
3. **The driver call itself isn't free.** Negligible compared to (1) and (2), but real.

*Where syncs hide.* Many ordinary operations imply one:

- **`.item()` / `.tolist()` / `.numpy()` / `float(loss)` / `print(loss)`** — pull a tensor's value into Python, runtime must wait for GPU.
- **`tensor.cpu()`** — copies to host; copy waits for the producing kernel.
- **A boolean check on a GPU tensor** — `if loss > threshold:` triggers a sync.
- **`assert` on a GPU tensor value** — same.
- **OOM / error reporting** — kernel error surfaces on the next CUDA call.
- **Transferring a non-pinned tensor with `non_blocking=True`** silently degrades to a sync.

Common surprise: "my training is slow because I added `print(loss.item())` every step." Yes — also added a per-step host↔device barrier.

*Why necessary anyway: timing.* Why TraceML uses CUDA events instead of `time.perf_counter`: perf_counter sees only **launch time**, not GPU completion. For host-side wall-clock measurement of GPU work, you must sync:

```python
import time, torch
torch.cuda.synchronize()
t0 = time.perf_counter()
y = model(x)
torch.cuda.synchronize()
t1 = time.perf_counter()
```

Two syncs bracket the GPU work — drain before, wait after. Cheapest correct way if you don't want to manage CUDA events. **CUDA events (`event.elapsed_time(other)`) are strictly better** for production timing — they live on the stream, observe the GPU without forcing the host to wait. TraceML uses events for that reason.

*What `synchronize()` is *not*.*

- **Not a barrier across processes.** Other ranks' GPUs unaffected. Cross-rank sync is what NCCL all-reduce does.
- **Not a memory fence.** Completion barrier.
- **Per-device, not global.** `torch.cuda.synchronize(device=1)` syncs only device 1.
- Does **not** flush the **caching allocator** (P33).

*Stream-level alternatives.*

- `stream.synchronize()` — wait for one stream's queue to drain. Cheaper than device-wide.
- `event.synchronize()` — wait for a specific recorded event. More surgical.
- `stream.wait_event(event)` — make stream B's future work wait for an event on stream A. **No host-side blocking** — wait enforced by GPU scheduler. Production async code uses the rightmost form.

*Why this matters for TraceML.* TraceML's bottleneck-finder value depends on **not** introducing per-step syncs. If timing instrumentation forced a `device.synchronize` every step, instrumented runs would be slower than uninstrumented runs and TraceML would distort what it measures. The CUDA event pool exists precisely so timing can be done without ever calling `synchronize()`.

When TraceML reports a layer is taking 12 ms of GPU time, that number was obtained by recording a CUDA event before and after the layer (both queued asynchronously on the same stream as the layer's kernels) and then, at some later moment when the events have fired, calling `event.elapsed_time(start, end)`. The host never blocked.

*Mental model.* `synchronize()` is `wait()` on every promise in a futures system — correct, sometimes necessary, lethal in a hot loop. Pipelining is the entire performance contract of async APIs; breaking pipelining is the canonical async anti-pattern.

**Concepts introduced:** `cudaDeviceSynchronize`, host↔device synchronization barrier, CPU/GPU pipelining and how it hides host overhead, kernel launch latency and queue refilling cost, implicit syncs (`.item()`, `print(tensor)`, `float(loss)`, boolean checks), per-device vs per-stream vs per-event sync granularity, `stream.wait_event` for GPU-side ordering without host blocking, why CUDA events beat `synchronize` + `perf_counter` for production timing, observer-effect risk in profiling.

---

### P33: How does PyTorch report GPU memory: `memory_allocated`, `max_memory_allocated`, the caching allocator?

**Date:** 2026-04-24

**Short answer:** `torch.cuda.memory_allocated()` reports bytes currently held by **live PyTorch tensors**. `torch.cuda.max_memory_allocated()` reports the **high-water mark** since last reset. Both are tracked by PyTorch's **caching allocator**, which sits between PyTorch and `cudaMalloc` — requests big slabs from the driver, then satisfies all subsequent tensor allocations from those slabs without going back. The number that matters for OOM is **`memory_reserved()`** (allocator-held), almost always larger than `memory_allocated()`.

**Long answer:**

*Why a caching allocator exists.* `cudaMalloc`/`cudaFree` are slow (hundreds of µs each) and implicitly synchronize with the device. A training step might allocate/free thousands of small intermediate tensors. If every tensor constructor called `cudaMalloc`, dispatch path would crawl. So PyTorch's caching allocator (`c10/cuda/CUDACachingAllocator.cpp`):

1. Asks `cudaMalloc` for **large blocks** (2 MB or 20 MB at a time).
2. Splits into smaller pieces for individual tensors.
3. On free, returns blocks to a **per-stream free list** instead of `cudaFree`.
4. Reuses freed blocks for the next allocation that fits.
5. Only calls `cudaFree` on specific triggers (`empty_cache`, OOM retry, rare conditions).

Result: alloc/free in a hot loop costs microseconds. The price: PyTorch's idea of "memory in use" diverges from the driver's.

*Two numbers.*

- **`torch.cuda.memory_allocated(device)`** — bytes currently held by live tensors. Goes up when a tensor is created, down on GC. Logical footprint of your program.
- **`torch.cuda.max_memory_allocated(device)`** — peak since last `reset_peak_memory_stats()`. What TraceML's step-memory sampler reports as "peak active during step N."
- **`torch.cuda.memory_reserved(device)`** (formerly `memory_cached`) — total bytes the allocator currently holds from the driver, including unused free-list blocks. What `cudaMalloc` told the driver, minus what's been given back.
- **`torch.cuda.max_memory_reserved(device)`** — peak of the above.

Inequality:

```
memory_allocated  <=  memory_reserved  <=  total_GPU_memory
```

The gap (`reserved - allocated`) is "free blocks the allocator is keeping for the next allocation." Intentional caching, not a bug.

*Worked example.*

```python
import torch
torch.cuda.reset_peak_memory_stats()
x = torch.empty(1024 * 1024 * 256, device="cuda")  # 1 GiB float32
print(torch.cuda.memory_allocated() / 2**30)       # ~1.0
print(torch.cuda.memory_reserved()  / 2**30)       # ~1.0

del x
print(torch.cuda.memory_allocated() / 2**30)       # ~0.0  (no live tensors)
print(torch.cuda.memory_reserved()  / 2**30)       # ~1.0  (allocator holds the block)

print(torch.cuda.max_memory_allocated() / 2**30)   # ~1.0
```

After `del x`, your tensor is gone — `memory_allocated` drops to zero. But the allocator did **not** call `cudaFree`; it kept the 1 GiB block on its free list. From the driver's perspective (and `nvidia-smi`'s), your process still owns 1 GiB.

*Reset APIs.* `torch.cuda.reset_peak_memory_stats()` zeros out the `max_*` counters but doesn't free anything. TraceML's step-memory sampler uses this pattern: reset at step start, sample peak at step end, ship the delta.

*`empty_cache()` — when it helps and when it hurts.* `torch.cuda.empty_cache()` walks the free list and `cudaFree`s every block with no live tensor. After this, `memory_reserved` shrinks toward `memory_allocated`.

When it **helps**: two GPU consumers in one process; about to invoke another GPU consumer; debugging OOM; after a giant temporary tensor.

When it **hurts**: inside the training loop. You destroy the free list; the next allocation re-calls `cudaMalloc` (slow + synchronizing). Forces `cudaFree`, which is itself synchronizing. Per-iteration latency goes up; "savings" are illusory because the allocator immediately rebuilds the free list. After the first epoch, the free list is exactly tuned for your workload — flushing it throws away that tuning.

The honest rule: do not call `empty_cache()` in normal training. Call it only at well-defined boundaries.

*Per-stream caching, briefly.* The allocator is **per-stream**: free lists keyed by which stream last used the block, because freed memory isn't safe to reuse until that stream's pending kernels are done. Correctness-preserving; most code uses default stream and never notices.

*Why this matters for TraceML.* TraceML reports `max_memory_allocated` per step — the peak live tensor footprint. Right number for "how big can my batch get before OOM." Not the right number for "how much GPU memory is my process holding" — for that, `memory_reserved`. Neither is correct for "how much does `nvidia-smi` see" (next question, P34).

A subtle gotcha: `memory_allocated()` is cheap (allocator-internal counter) but on some paths acquires a lock. Hammering it inside a hot per-op hook can show up in profiles. TraceML samples it at sampler-tick frequency (1 Hz default) and at step boundaries.

**Concepts introduced:** caching allocator, why `cudaMalloc`/`cudaFree` are expensive and synchronizing, large-block strategy with split-and-recombine, per-stream free lists, `memory_allocated` vs `memory_reserved`, `max_*` peak counters and `reset_peak_memory_stats`, `memory_stats()` for deep debugging, `empty_cache()` semantics, intentional gap between allocated and reserved.

---

### P34: Why is reported memory sometimes lower than `nvidia-smi` shows? (caching allocator, fragmentation)

**Date:** 2026-04-24

**Short answer:** `nvidia-smi` shows what the **NVIDIA driver** thinks a process owns — every byte of every block ever obtained via `cudaMalloc` and not yet returned, **plus** the **CUDA context's** own overhead (kernels, libraries, workspaces; see [Q11](learning-qa.md#q11-what-is-a-cuda-context-and-why-is-it-fork-unsafe)). PyTorch's `memory_allocated()` shows only what live tensors hold. The gap: (1) the **caching allocator** keeping freed blocks for reuse, (2) **CUDA context overhead** (often 300 MB–1 GB just from loaded kernels), and (3) **fragmentation** — blocks the allocator owns but can't satisfy a request from. `nvidia-smi` is always >= `memory_reserved()` >= `memory_allocated()`.

**Long answer:**

*The hierarchy of "memory used" numbers.*

```
memory_allocated   <=  memory_reserved   <=  nvidia-smi (process)   <=  total GPU memory
   (live tensors)      (allocator-held)      (driver-attributed)
```

Each gap has a name and a cause:

1. **`memory_reserved` − `memory_allocated`** — caching allocator's free list (P33). Blocks PyTorch grabbed, used, freed, but didn't return.
2. **`nvidia-smi` − `memory_reserved`** — **CUDA context overhead**. Memory the driver attributes to your process that PyTorch's allocator never sees:
   - **Loaded kernels and CUDA libraries** — cuBLAS, cuDNN, possibly NCCL, plus thousands of compiled PyTorch kernels into GPU instruction memory. 200-800 MB depending on what's touched.
   - **CUDA context bookkeeping** — per-context tables, command queues, pinned-memory allocations.
   - **Driver workspaces** — cuBLAS and cuDNN keep persistent scratch (hundreds of MB for cuDNN convolution algorithms alone).
   - **NCCL communication buffers** (in distributed runs) — several hundred MB per process.

   None of this is allocated through `cudaMalloc` in a way the caching allocator tracks. From `import torch; torch.zeros(1, device="cuda")` you typically see 400-1000 MB of `nvidia-smi` usage with `memory_allocated` near zero.

3. **Total − nvidia-smi** — actual free GPU memory the driver could give to *any* process.

Practical example: train ResNet-50 on an A100. `nvidia-smi`: 8 GB. `memory_reserved`: 6.5 GB. `memory_allocated`: 5.8 GB peak / 1.5 GB idle. That's: 1.5 GB live tensors at idle, +4.3 GB during step's peak, +0.7 GB cached free blocks, +1.5 GB CUDA context. All four correct; different questions.

*Fragmentation: the allocator owns memory it can't use.* Even within `memory_reserved`, not every byte is usable. Suppose the allocator holds a 1 GiB segment split over time:

```
[ 200 MB live ][ 100 MB free ][ 300 MB live ][ 100 MB free ][ 300 MB live ][ 24 MB free ]
```

Total free: 224 MB. Largest **contiguous** free piece: 100 MB. A request for 150 MB will fail with **OOM despite 224 MB being free**. PyTorch will (depending on settings) try to split a larger segment, ask the driver for more memory, retry after `empty_cache`, or give up:

```
CUDA out of memory. Tried to allocate 150.00 MiB.
GPU 0 has a total capacity of 40.00 GiB of which 200.00 MiB is free.
Of the allocated memory, 35.50 GiB is allocated by PyTorch, and
3.80 GiB is reserved by PyTorch but unallocated.
```

`reserved but unallocated` is the fragmentation gap.

*What causes fragmentation.*

- **Variable batch sizes / sequence lengths** — common in NLP and RL. Free-list shape tuned to recent allocations; mixing sizes leaves gaps.
- **Activation checkpointing combined with normal forward** — different lifetime patterns thrash the allocator.
- **Loading and unloading models** mid-job.
- **Long-lived tiny tensors scattered through big blocks** — pin large segments by occupying small pieces.

*How PyTorch fights fragmentation.* `PYTORCH_CUDA_ALLOC_CONF`:

- `max_split_size_mb:N` — never split a free block larger than N MB.
- `expandable_segments:True` (newer) — uses **virtual memory** tricks to grow segments contiguously without re-fragmenting. Dramatically reduces fragmentation for variable-shape workloads.
- `garbage_collection_threshold:0.8` — triggers a GC pass when reserved exceeds this fraction.

Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is currently the highest-leverage fix for fragmentation-driven OOMs.

*Why disagreement is normal.* Driver knows how much GPU memory your process sits on. PyTorch knows how much its allocator holds. Application knows how much its tensors hold. Three different layers, real overhead between each.

| Question                                            | Number                  |
| --------------------------------------------------- | ----------------------- |
| Will my next allocation fit?                        | `memory_reserved` vs request |
| How big can my batch be in steady state?            | `max_memory_allocated`  |
| How much memory blocks others?                       | `nvidia-smi` for process |
| Is allocator wasting memory on cached blocks?       | `reserved - allocated`  |
| Am I OOM because of fragmentation?                  | `largest_free_block` in `memory_stats` vs request |

*Multi-process pitfalls.* Two PyTorch processes on one GPU each see their own caching allocator; they don't share blocks. Both processes' contexts overhead is paid twice. **MPS (Multi-Process Service)** lets multiple processes share one CUDA context to amortize.

*Why this matters for TraceML.* TraceML reports PyTorch-side numbers. When a user says "TraceML says 5 GB but `nvidia-smi` shows 8 GB":
- ~700 MB CUDA context + cuDNN/cuBLAS workspaces (one-time baseline).
- ~1.5 GB allocator free list.
- ~700 MB NCCL buffers if DDP.
- All correct; different layers.

Future TraceML enhancements that touch memory advisory should report all four numbers (live, peak, reserved, driver-visible) and largest-free-block — only honest set.

*Mental model.* Like Python's `gc.get_objects()` vs `RSS`: gc shows live objects; RSS shows what OS gave the interpreter; the gap is allocator caching, interpreter overhead, loaded `.so` files, arena fragmentation. Same pattern, on the GPU. PyTorch is the interpreter; the caching allocator is malloc; CUDA context overhead is `.so` loading; `nvidia-smi` is `top`.

**Concepts introduced:** the four memory numbers (`allocated`, `reserved`, `nvidia-smi`, total), CUDA context overhead, fragmentation as "free but not contiguous," OOM message anatomy, `PYTORCH_CUDA_ALLOC_CONF` knobs (`max_split_size_mb`, `expandable_segments`, `garbage_collection_threshold`), `expandable_segments` virtual-memory trick, MPS for shared CUDA contexts, RSS-vs-gc analogy.

---

## Mixed precision (AMP)

### P35: What does `torch.autocast` actually wrap, and how does it pick which ops to downcast?

**Date:** 2026-04-24

**Short answer:** `torch.autocast` is a **context manager that pushes a thread-local dispatch key** (`Autocast` / `AutocastCUDA`) onto PyTorch's dispatcher stack. While that key is active, every op routed through the dispatcher consults a **per-op autocast policy** — an allowlist of ops to downcast to fp16/bf16 (matmul-heavy: `mm`, `bmm`, `addmm`, `linear`, `conv*`, `*_attention`), a blocklist that stays in fp32 (reductions and numerically sensitive ops: `softmax`, `log`, `exp`, `pow`, loss functions, `layer_norm`, `batch_norm`), and a "promote" set where mixed-dtype inputs get widened to fp32. No model rewriting, no graph capture, no monkey-patching — a dispatcher key plus a policy table baked into C++.

**Long answer:**

*The dispatcher hook.* From P30/P31, PyTorch routes every op through a stack of dispatch keys. `torch.autocast("cuda", dtype=torch.float16)` enables the `AutocastCUDA` key on the current thread (and disables on `__exit__`). The Autocast key sits *above* the backend key, so it intercepts the call before the actual CUDA kernel runs. Does its dtype rewriting, then **redispatches** — falls through to the next key. Same redispatch pattern Autograd uses to wrap forward ops with grad recording.

Thread-local dispatch key on a stack: nested `with autocast(enabled=False):` flips it off for that block; other threads unaffected. No global state.

*The policy table.* Per-op lookup, defined in C++ (`aten/src/ATen/autocast_mode.cpp`). Each op registered into one of four buckets:

| Bucket | Behavior | Examples |
|---|---|---|
| `lower_precision_fp` | Cast all floating inputs to autocast dtype | `mm`, `bmm`, `addmm`, `linear`, `conv1d/2d/3d`, `prelu`, `_scaled_dot_product_*_attention`, `matmul` |
| `fp32` | Force inputs to fp32 | `softmax`, `log_softmax`, `nll_loss`, `cross_entropy`, `binary_cross_entropy`, `layer_norm`, `batch_norm`, `pow`, `log`, `exp`, `cumsum`, `prod` |
| `fp32_set_opt_dtype` | Run in fp32 unless explicit `dtype` arg overrides | reductions where user can opt in |
| `promote` | If mixed dtypes, widen to widest input | `cat`, `stack`, `addcdiv`, `addcmul`, `index_put`, comparison ops |

Anything not registered: autocast redispatches without touching dtypes. So `view`, `reshape`, indexing, copies are dtype-transparent.

The split is not arbitrary. Matmuls and convs go in allowlist because they (a) dominate FLOPs, (b) accumulate into fp32 internally on Tensor Cores. Reductions and pointwise transcendentals go in blocklist because their accumulators are software-visible — `softmax` over 50k tokens in fp16 will overflow `exp` for any logit ≥ ~11. `layer_norm`/`batch_norm` blocklisted because variance computation is catastrophically lossy in fp16.

*Concrete walkthrough.*

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    h = self.linear(x)          # x is fp32
    h = torch.nn.functional.relu(h)
    logits = self.head(h)
    loss = F.cross_entropy(logits, targets)
loss.backward()
```

1. `self.linear(x)` → `aten::linear`. Autocast sees `linear` is `lower_precision_fp`, casts `x` and weight/bias to fp16, redispatches. CUDA kernel runs in fp16 with fp32 accumulation. Output `h` is fp16.
2. `relu(h)` — not in any bucket. Redispatches unchanged. Output stays fp16.
3. `self.head(h)` — same as 1. Output `logits` is fp16.
4. `F.cross_entropy(logits, targets)` → internally `log_softmax` + `nll_loss`, both `fp32` bucket. Autocast casts `logits` *back up* to fp32. Output `loss` is fp32.

*Weights are not changed.* Your `Linear.weight` parameter is still fp32 in storage. Autocast casts a fp16 *copy* on each forward call, on the fly. That copy is cached for the duration of the autocast region (so back-to-back `linear` calls reusing the same weight don't pay the cast twice), flushed on `__exit__`. This is why "AMP doesn't change your model" — params and grads on disk and in optimizer state remain fp32.

*Backward.* The autocast region typically ends at `loss.backward()` — backward runs *outside* autocast context. But the autograd engine remembers, per saved-tensor, what dtype each forward op produced; the corresponding backward kernel runs in same precision. fp16 `mm` forward records a backward that does fp16 `mm`s for grad-input and grad-weight, accumulating into fp32 on Tensor Cores. The grads landing in `param.grad` are then in fp16 (for cast weights). This is why fp16 needs `GradScaler` (P36).

*Things that surprise people.*

- **Per-thread.** A `DataLoader` worker doesn't inherit it. Doesn't matter (workers don't run model ops), but matters if you spin custom threads.
- **Only intercepts ops registered into ATen.** A custom CUDA kernel via `torch.utils.cpp_extension` won't be autocast-aware unless registered into the policy.
- **`@torch.cuda.amp.custom_fwd(cast_inputs=...)`** is the user-facing API to opt a custom autograd Function into the policy.
- **Doesn't cast Python scalars or non-tensor args.** Only floating-point tensor inputs.
- **`device_type="cuda"` and `device_type="cpu"` are separate keys** with separate policy tables.

*Mental model.* Autocast is a thin **policy layer between the user-facing op and the kernel**. Doesn't transform your model. Doesn't look at your computation graph. Sees one op at a time, asks "which bucket?", maybe casts inputs, forwards. The cleverness is in the choice of buckets — hand-curated based on numerical analysis.

**Concepts introduced:** `torch.autocast` as thread-local dispatch key, `AutocastCUDA` / `AutocastCPU` keys, autocast op policy buckets, Tensor Core fp32 accumulation, on-the-fly weight cast caching, weights-stay-fp32 invariant, autocast composes via dispatcher redispatch, autograd remembers per-tensor dtype, `custom_fwd` for user kernels.

---

### P36: Why does fp16 need a `GradScaler`, but bf16 doesn't?

**Date:** 2026-04-24

**Short answer:** fp16 has a **5-bit exponent** — its smallest normal positive value is ~6e-5, so any gradient component below that **underflows to zero** before reaching the optimizer. Many real gradients live below 1e-5. `GradScaler` works around this by **multiplying the loss by a large constant** (typically 2^16) before backward, shifting the entire gradient distribution into fp16's representable range, then **dividing the gradients back down** in fp32 before the optimizer step. bf16 has the same **8-bit exponent as fp32** — gradients fit natively. Cost: bf16 has only 7 bits of mantissa vs fp16's 10, so it's "wider but coarser."

**Long answer:**

*The underflow problem.* Smallest positive normal fp16: `2^-14 ≈ 6.1e-5`. Smallest positive subnormal: `2^-24 ≈ 6e-8`. A weight-grad component of `3e-7` rounds to **zero** when stored in fp16. Now imagine a fp16 forward → fp16 backward on a 7B-parameter model: a meaningful chunk of `param.grad` entries are exactly 0 not because the math said so but because they couldn't be represented. Optimizer sees zero grad → no update → those weights freeze. Training silently regresses.

bf16 has `2^-126 ≈ 1.2e-38` as its smallest normal — same as fp32. The gradient distribution that fits in fp32 also fits in bf16. No underflow.

*What `GradScaler` does.*

```python
scaler = torch.cuda.amp.GradScaler()  # initial scale = 2^16 = 65536

for batch in loader:
    optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(batch)
    scaler.scale(loss).backward()      # backward on (loss * S), grads are S× larger
    scaler.unscale_(optimizer)         # divide grads by S, in fp32, in-place
    torch.nn.utils.clip_grad_norm_(...)# now safe to inspect/clip true grads
    scaler.step(optimizer)             # checks for inf/nan; if clean, calls step()
    scaler.update()                    # adapt scale factor for next iter
```

Three pieces of magic:

**1. Scaling the loss.** Multiplying loss by `S` is mathematically equivalent (chain rule) to multiplying every grad by `S`. So `scaler.scale(loss).backward()` produces grads `S` times bigger — pulling the underflow tail into representable range.

**2. Unscaling before optimizer.** Before `optimizer.step()` sees grads, divide back by `S`. `scaler.unscale_(optimizer)` casts grads to fp32, divides by `S` in place. After this, `param.grad` holds true grad in fp32. Only need to call `unscale_` explicitly to inspect/clip; `scaler.step` does it implicitly.

**3. Inf/NaN check + skip.** While unscaling, scaler scans for `inf`/`NaN`. If found:
- **Skips the optimizer step** (`optimizer.step()` not called).
- Halves `S` for next iteration via `scaler.update()`.

If consecutive iterations succeed cleanly (default 2000), `scaler.update()` *doubles* `S`. So `S` is adaptive — finds the largest scale your model can tolerate without overflowing.

*Why the skip is correct.* An overflow means a grad went to `+inf` during backward — either true grad really is enormous (bad init) or scale was too aggressive. Applying that step would corrupt weights. Skipping: scale reduced, next iteration retries on a fresh batch. Negligible cost vs corrupted weight matrix.

*Skipped steps in practice.* Every well-instrumented AMP loop sees a flurry of skipped steps in the first ~50 iterations as `S` settles. After: rare (<1 in 1000). If skips persist deep into training, it's a signal.

*bf16 simplicity.*

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(batch)
loss.backward()                     # no scaler
optimizer.step()
```

That's it. No skip logic, no scale tuning, no `scaler.update()`. Grads are bf16-valued, optimizer reads them, optimizer state stays fp32, update is fine.

*The price you pay for bf16.* bf16 mantissa is 7 bits (vs fp16's 10, fp32's 23):

- **Coarser quantization.** Adjacent bf16 values differ by ~0.4% (`2^-7`). Adjacent fp16 by ~0.1% (`2^-10`).
- **Worse for stable inner-product accumulation that *doesn't* use Tensor Core fp32 accumulators.** Most matmuls are fine because hardware accumulates in fp32 internally. But explicit reduction loops in user code lose precision faster in bf16 than fp16.

For LLM training the bf16 tradeoff is overwhelmingly favored — range matters more than resolution because gradients span many orders of magnitude. For some classical CV models with tight numerical envelopes, fp16 + GradScaler can give better final accuracy.

*Hardware availability.* fp16 has been on every NVIDIA GPU since Volta (V100, 2017). bf16 needs Ampere (A100, 2020) or newer for native Tensor Core support. So "just use bf16" is a 2020s answer.

*TraceML interaction.* If you see `optimizer.step` time bimodally distributed (some fast, some slow), the fast ones may be skipped iterations — `scaler.step(optimizer)` short-circuits.

*Mental model.* fp16 is a **narrow window**: high precision but small range. The scaler is a **periscope** that slides the window over the grad distribution. bf16 is a **wide window with grainy glass**: covers all magnitudes but with less detail. AMP is your choice of compromise.

**Concepts introduced:** fp16 underflow at `2^-14`, bf16 having fp32's exponent range, `GradScaler` as loss-scaling + adaptive scale + inf/NaN skip, scale adaptation (halving on overflow, doubling after N clean steps), `scaler.scale` / `scaler.unscale_` / `scaler.step` / `scaler.update` lifecycle, mantissa precision tradeoff (7 vs 10 bits), Tensor Core fp32 accumulation, hardware availability gating (Volta vs Ampere), skipped-step bimodal step-time signature.

---

### P37: When does AMP cause NaN losses or silent precision loss, and how do you debug it?

**Date:** 2026-04-24

**Short answer:** NaN/Inf loss under AMP comes from a small set of mechanisms: **fp16 overflow in an op the autocast policy didn't catch**, **fp16 underflow zeroing grads** before `GradScaler` settles, **a custom CUDA op or `torch.autograd.Function` not registered with the autocast policy**, or a **numerically unsafe pattern** (large softmax, manual log/exp, `1/x` near zero). Silent precision loss usually traces to **operations downcast that shouldn't have been**, **bf16 reductions accumulating drift**, or **per-step grad noise from aggressive `GradScaler` settings**. Debug by isolating: bisect with `torch.autograd.set_detect_anomaly(True)`, force suspect regions to fp32, log per-layer activation/grad min/max/has_nan, compare against fp32 baseline at fixed seed.

**Long answer:**

*Five common failure modes.*

**1. fp16 overflow in a permitted op.** A poorly-initialized linear layer producing logits of magnitude `1e5` overflows on the forward pass (fp16 max ~6.5e4). Symptom: `loss = nan` from iteration 1, before `GradScaler` reacts. Fix: rescale init, add normalization, or force that layer to fp32.

**2. fp16 underflow at start of training.** Before `GradScaler` adapts (~50 steps), `S` is at initial value (2^16). If true grad magnitude is so small that even `65536 × grad` underflows, every iteration "succeeds" but writes zeros to `param.grad`. Loss stagnates, doesn't NaN. Fix: higher `init_scale=2**24`, or check `scaler.get_scale()` over iterations.

**3. Custom op outside autocast policy.** If you wrote a `torch.autograd.Function` for a custom kernel and didn't decorate with `@torch.cuda.amp.custom_fwd`, autocast doesn't know what to do. Inputs arrive in whatever dtype the previous op produced (often fp16); your kernel does fp16 math you didn't anticipate. Symptom: NaNs only when this op is in graph. Fix: `@custom_fwd(cast_inputs=torch.float32)` and `@custom_bwd`.

**4. Numerically unsafe pattern not in blocklist.** Autocast blocklists obvious things but can't catch user-written equivalents:

```python
# innocent looking, explodes in fp16
attn_weights = (q @ k.transpose(-2, -1))     # fp16 matmul -- ok
attn_weights = attn_weights.softmax(dim=-1)  # autocast forces fp32 -- ok
y = attn_weights @ v                          # fp16 matmul -- ok

# user-written variant blows up:
attn_weights = (q @ k.transpose(-2, -1)).exp()   # exp on fp16 logits!
attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
```

Manual `exp + normalize` isn't recognized as softmax, so isn't blocklisted. fp16 `exp(11) ≈ inf`. NaN cascade. Fix: use `F.softmax` or wrap in `with torch.autocast(enabled=False):`. Same trap with manual `log_softmax`, manual cross-entropy, manual variance.

**5. Loss-domain math at model output.** Model returns fp16; you compute custom loss outside autocast. `(pred - target).pow(4).mean()` in fp16 overflows for `|pred - target| > ~16`. Move loss inside autocast (so `pow` gets blocklisted) or cast `pred` to fp32 first.

*Silent precision loss.*

**bf16 reduction drift.** Long sum or mean in bf16 with 7-bit mantissa loses bits every few thousand additions. `tensor.mean()` over a million elements in bf16 ends up off by a few percent from fp32. Hidden by Tensor Core fp32 accumulation in matmuls; not hidden in explicit `.sum()`/`.mean()` chains, EMA buffers, hand-rolled layernorm-equivalents.

**Aggressive `GradScaler` settings.** If `growth_factor` too high or `growth_interval` too short, `S` keeps pushing into territory where occasional overflow happens. Each → skipped step → progress lost. Aggregate: optimizer sees fewer real updates than your epoch counter implies.

**Layer-norm bf16 drift.** Even though `layer_norm` is in fp32 blocklist for autocast, if you cast the whole model to bf16 (`.bfloat16()`), the blocklist is bypassed (autocast not active — params are simply bf16). Layernorm variance in bf16 is unstable for large hidden sizes. Fix: keep layernorm parameters in fp32 explicitly.

*The debugging flow.*

**Step 1: confirm AMP is the cause.** Disable autocast and rerun. If NaN goes away, AMP is involved.

**Step 2: pin the producing op.** Anomaly mode:

```python
with torch.autograd.set_detect_anomaly(True):
    loss = model(batch)
    loss.backward()
```

Makes autograd raise at the op that produced the NaN, with stack trace. 10–50× slowdown — use on a single batch.

**Step 3: bisect by layer.** Wrap suspect segments in `with torch.autocast(enabled=False):`. If NaN goes away, bug is in that segment. Common transformer culprits: attention numerator (pre-softmax logits), residual stream after several layers, output head.

**Step 4: log min/max/has_nan per layer.** Forward hooks (see [Q9](learning-qa.md#q9-what-are-hooks-and-what-does-injecting-hooks-in-process-mean)):

```python
def watch(name):
    def hook(module, inp, out):
        if torch.is_tensor(out):
            t = out.detach()
            print(f"{name}: dtype={t.dtype} "
                  f"min={t.min().item():.3e} max={t.max().item():.3e} "
                  f"nan={torch.isnan(t).any().item()} "
                  f"inf={torch.isinf(t).any().item()}")
    return hook

for name, module in model.named_modules():
    module.register_forward_hook(watch(name))
```

The first layer whose `max` jumps to `~6e4` (fp16) or `~3e38` (bf16) or whose `nan=True` flips is your suspect.

**Step 5: check `GradScaler` health.** Log `scaler.get_scale()` and skip count. Healthy: scale settles within 50 steps to `2^7`–`2^16`, <1% of steps skipped. Unhealthy: oscillating, or sits at `1.0` (model overflows even at no scaling).

**Step 6: compare against fp32 baseline at fixed seed.** Run 100 iterations of fp32, 100 of AMP, same seed and data order. Losses should match to ~3 decimals for first iteration and diverge slowly. Fast divergence (>5% by iter 10) = AMP is changing the math more than expected.

*When to switch from fp16 to bf16.* If you've spent more than a day chasing fp16 NaNs and hardware supports bf16 (Ampere+), switch. Cost of bf16's coarser mantissa is almost always less than engineering cost of debugging fp16 for a model that doesn't naturally fit. fp16+scaler is right for inference deployment and older hardware; bf16 is right for new training.

*TraceML potential.* Per-layer activation min/max would be a natural extension to surface "this is the layer where activations overflow under AMP." Currently you'd diagnose with the manual hook approach.

*Mental model.* AMP is **lossy compression of the math**. Works when the loss profile is unchanged (compression in noise floor). Breaks when compression hits a real signal — usually because an op needing precision didn't get blocklisted, or an op fitting in fp32's range doesn't fit in fp16's. Debugging AMP is the same skill as debugging quantization.

**Concepts introduced:** fp16 overflow vs underflow failure modes, `GradScaler` warm-up zero-grad failure, custom op autocast registration via `custom_fwd`/`custom_bwd`, manual softmax/log_softmax as autocast trap, loss-domain overflow at model output, bf16 reduction drift, layernorm-in-bf16-without-autocast pitfall, `set_detect_anomaly` for op-level NaN attribution, layer bisection with nested `autocast(enabled=False)`, forward-hook NaN/Inf monitoring, `scaler.get_scale()` health signal, fixed-seed AMP-vs-fp32 baseline comparison.

---

## Distributed (DDP / FSDP)

### P38: What does `torch.distributed.init_process_group()` actually set up?

**Date:** 2026-04-24

**Short answer:** `init_process_group` is the call that turns N independently-launched Python processes into a coordinated **process group** that can issue collective communications. It picks a **backend** (NCCL on GPU, Gloo on CPU), uses a **rendezvous mechanism** (TCP store, file, or env vars) so every rank discovers every other rank's address, and constructs the in-driver communicator handles needed for collectives. After it returns, every rank knows its own [rank](learning-qa.md#q4-what-is-a-gpu-rank), the world size, and how to talk to peers.

**Long answer:**

*What "process group" means.* PyTorch distributed is built on the abstraction: a fixed set of processes that have collectively agreed they will participate in collective ops together. Calling `dist.all_reduce(tensor)` makes no sense in isolation — every rank in the group must call it with a compatible tensor, and the collective is the synchronous combination. `init_process_group` creates this shared abstraction across processes that were launched independently and don't know about each other yet.

*The launch precondition.* Before `init_process_group` runs, you've already started N processes (typically via `torchrun` or `torch.multiprocessing.spawn`). Each process has env vars: `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `LOCAL_RANK`. Nothing in the cluster is connected yet — just N CPython interpreters with matching env vars.

*Step 1: rendezvous.* `init_process_group` first runs a **rendezvous** so every rank discovers every other's address. Default `init_method="env://"` reads `MASTER_ADDR:MASTER_PORT` and connects to a `TCPStore` running on rank 0 (rank 0 binds the port; ranks 1..N-1 connect as clients). Each rank writes its address into the store; everyone reads everyone else's. After this barrier, every rank has the full address book. Alternative: `init_method="file://..."` for shared filesystem rendezvous. See [Q10](learning-qa.md#q10-what-is-tcp-concretely-and-whats-a-port).

*Step 2: backend initialization.* With addresses known, the chosen **backend** initializes peer-to-peer communicators:

- **NCCL** (GPU): each rank calls `ncclCommInitRank`, which exchanges unique IDs via the store, opens **NCCL channels** between every pair of GPUs that will participate, sets up shared-memory regions for intra-node peers, and (if available) opens RDMA queue pairs over Infiniband — see [Q14](learning-qa.md#q14-what-is-rdma-infiniband-and-why-does-it-matter-for-multi-node-training). After this, the NCCL communicator is a heavy in-driver object holding GPU buffers, CUDA streams, per-peer transport state. This is also why `init_process_group` is slow at scale — pairwise transports take seconds.
- **Gloo** (CPU): pure TCP-based collectives, simpler, used for CPU tensors or small control messages.
- **MPI**: thin wrapper over an installed MPI implementation; rare in pure PyTorch workflows.

*Step 3: register the default group.* The newly built group becomes the **default process group**. Subsequent `dist.all_reduce(t)` (without `group=`) operates on it. You can build sub-groups later with `dist.new_group([0, 2, 4])` for partial collectives — useful in tensor-parallel topologies.

*What you have after the call.*

- `dist.get_rank()` returns this process's global rank.
- `dist.get_world_size()` returns N.
- `dist.is_initialized()` is True.
- The CUDA context on each GPU now contains an NCCL communicator with open channels to all peers.
- A `barrier()` between ranks is now possible.

*Cost and ordering pitfalls.*

- `init_process_group` itself is a barrier — every rank blocks until everyone has joined. If one rank crashes before calling it, everyone else hangs. First place a misconfigured cluster reveals itself.
- Call `torch.cuda.set_device(local_rank)` *before* `init_process_group` for NCCL. Otherwise NCCL picks the wrong GPU and you get cryptic "duplicate GPU" errors.
- NCCL touches CUDA — this call initializes a CUDA context. Forking after this point is unsafe ([Q11](learning-qa.md#q11-what-is-a-cuda-context-and-why-is-it-fork-unsafe)).

*Mental model.* `init_process_group` is like opening a multiplayer game lobby. Everyone joined with their handle (rank), the lobby figured out who's at which IP, the netcode allocated buffers, and after the handshake the game can issue synchronized actions ("everyone fire!" = all-reduce). What actually happens during a collective is its own story — see [Q12](learning-qa.md#q12-what-is-nccl-all-reduce-and-why-is-it-a-barrier).

**Concepts introduced:** process group, rendezvous, `TCPStore`, `init_method` (env/file/store), backend (NCCL / Gloo / MPI), `MASTER_ADDR` / `MASTER_PORT`, `ncclCommInitRank`, NCCL channels, default process group, `new_group` for sub-groups, init-time barrier semantics, `set_device` ordering requirement.

---

### P39: What does `DistributedDataParallel` wrap around your model? How is gradient sync inserted into backward?

**Date:** 2026-04-24

**Short answer:** `DistributedDataParallel` (DDP) wraps your `nn.Module` in a thin Python wrapper, but the real machinery is a C++ object called the **Reducer**. At construction, the Reducer registers an **autograd accumulation hook** on every parameter's `.grad` accumulator node. As backward runs, each hook fires the moment a parameter's gradient is finalized, and the Reducer copies that gradient into a **bucket**. When a bucket fills, the Reducer launches an asynchronous **all-reduce** on it on a dedicated NCCL stream. By the time backward finishes, most all-reduces are already complete because they overlapped with the rest of backward.

**Long answer:**

*The wrapper layer.* `model = DDP(model, device_ids=[local_rank])` returns a wrapper module. Its `forward` does almost nothing interesting — calls the inner module's forward (with bookkeeping for unused parameters and replicated streams). The Python wrapper is a thin shell. At construction, DDP also broadcasts rank 0's parameters to all other ranks so everyone starts from the same weights — why you don't need to manually seed weight init across ranks.

*The real work at construction.* Inside `DDP.__init__`, after parameter broadcast, DDP builds a **Reducer** (C++ class in `torch/csrc/distributed/c10d/reducer.cpp`). The Reducer (a) owns the all-reduce schedule, (b) inserts hooks into autograd for notification, (c) batches gradients into buckets so one large all-reduce is issued instead of many tiny ones.

*How the hook plugs into autograd.* Every leaf tensor with `requires_grad=True` has, in the autograd graph, an `AccumulateGrad` node — that's the node that takes incoming grad and writes it into `param.grad`. The Reducer walks all model parameters and registers a **post-accumulation hook** on the AccumulateGrad node:

```python
# pseudocode of what DDP does at init
for p in model.parameters():
    if p.requires_grad:
        grad_acc = p.grad_accumulator()  # the AccumulateGrad node
        grad_acc.register_post_hook(
            lambda *_: reducer.mark_variable_ready(p)
        )
```

Same hook concept covered in [Q9](learning-qa.md#q9-what-are-hooks-and-what-does-injecting-hooks-in-process-mean), on autograd graph nodes rather than `nn.Module` instances. The hook fires inside the autograd engine, immediately after `param.grad` gets its final value. From the user's perspective backward looks normal; the Reducer is an invisible co-process.

*Bucketing.* If DDP issued one all-reduce per parameter, a 1000-layer model would issue 1000 separate NCCL collectives — each with kernel launch overhead, latency, pipeline bubble. Instead, the Reducer groups consecutive parameters into **buckets** of fixed byte size (default 25 MB; tunable via `bucket_cap_mb`, P40). When `mark_variable_ready` is called for the last parameter in a bucket, the Reducer fires an **async** `all_reduce` on the bucket's flat buffer.

*Why "reverse of forward" order.* Backward processes layers from loss back to input. Last layer's gradients ready first; first layer's last. DDP knows this and assigns parameters to buckets in **reverse construction order** (by default), so the first bucket to fill corresponds to layers near the output. That bucket can start its all-reduce while the rest of backward is still running. By the time input-side layers finish backward, several buckets' all-reduces are already in flight.

*Stream-level overlap.* The all-reduce isn't just async at the Python level — it runs on a **dedicated NCCL stream** distinct from the compute stream. See [Q15](learning-qa.md#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread). Compute stream keeps running backward kernels; comm stream runs NCCL ring-reduce in parallel on the same GPU's SMs and on the network. CUDA events synchronize the two streams. This **compute-comm overlap** is the entire performance story of DDP — well-tuned DDP gets near-linear scaling on networks fast enough to keep up with backward.

*The finalize barrier.* DDP installs a final hook (via `prepare_for_backward` and `finalize_backward`) ensuring the optimizer step doesn't read `param.grad` before all bucket all-reduces have completed. So even though all-reduces run async, `optimizer.step()` always sees fully averaged gradients.

*`register_comm_hook` — user-replaceable comm.* DDP exposes `model.register_comm_hook(state, hook)` so users can replace the default all-reduce with custom comm. Built-in hooks: `fp16_compress_hook` (cast to fp16 before all-reduce, halves bandwidth), `powerSGD_hook` (low-rank gradient compression). The hook receives a `GradBucket` and returns a future that resolves to the reduced tensor.

*Gotcha: unused parameters.* If forward is dynamic (control-flow branch skips a layer), some parameters won't get gradients. The Reducer waits forever for those buckets → deadlock. Fix: `DDP(..., find_unused_parameters=True)` traces the autograd graph at the start of backward and proactively marks unreachable parameters as ready. Costs ~5–10% of step time; leave off when not needed.

*Where TraceML fits.* DDP's hook injection and bucket scheduling happen alongside TraceML's own hook injection (Q9). TraceML measures per-layer forward/backward time; DDP measures gradient bytes and triggers comm. Both are passive observers; they don't conflict because they hook at different points (TraceML on `nn.Module`, DDP on `AccumulateGrad`). When TraceML reports "backward time" much larger than the sum of per-layer backward times, the difference is largely waiting for the DDP all-reduce buckets to drain at the end of backward.

*Mental model.* The Reducer is an actor sitting next to the autograd engine. Every time a parameter shouts "I'm done!", the Reducer notes it and dumps the gradient into the next bucket. When a bucket is full, it slings the bucket onto the conveyor belt to NCCL (the comm stream), which carries it over the network for averaging. The whole thing is a producer-consumer pipeline running concurrently with backward.

**Concepts introduced:** DDP wrapper vs Reducer, `AccumulateGrad` node, post-accumulation autograd hooks, gradient bucketing, async all-reduce, dedicated NCCL stream, compute-comm overlap, reverse-order bucket fill, `find_unused_parameters`, `register_comm_hook` and built-in compression hooks (fp16, PowerSGD), broadcast-at-init for parameter consistency, the finalize barrier before optimizer step.

---

### P40: What is gradient bucketing, and what does `bucket_cap_mb` tune?

**Date:** 2026-04-24

**Short answer:** **Gradient bucketing** is DDP's strategy of concatenating many small gradient tensors into a single contiguous byte buffer (a "bucket"), then issuing one all-reduce on the whole buffer instead of one per parameter. `bucket_cap_mb` sets the target byte size of each bucket (default 25 MB). Smaller buckets give finer-grained overlap with backward but more NCCL launches and lower bandwidth utilization; larger buckets amortize launch overhead and saturate the network better but delay the start of all-reduce until more of backward has completed.

**Long answer:**

*Why bucketing exists.* A modern transformer has thousands of parameters of wildly different sizes. If you all-reduce each one independently you pay:

- A NCCL kernel-launch cost (~10s of µs each) per parameter.
- A latency floor per collective (network RTT, ring traversal startup) — bandwidth doesn't matter until message is large enough to fill the pipeline.
- Loss of pipelining: small messages don't overlap meaningfully.

For small tensors, latency dominates. Bucketing flips that: pack many small grads into a 25 MB buffer, pay one launch cost, the all-reduce is bandwidth-limited (the regime where NCCL is efficient).

*How a bucket is built.* The Reducer (P39) decides at construction time which parameters go into which bucket. Each bucket allocates one flat contiguous tensor sized `bucket_cap_mb` MB. As each parameter's `AccumulateGrad` hook fires, its gradient is **copied** into its slice of the bucket buffer. When all parameters in a bucket have reported in, the Reducer launches `ncclAllReduce(bucket_buffer)`. After return, the Reducer copies averaged data back out into each `param.grad` (or makes parameters into views via `gradient_as_bucket_view=True` to avoid the second copy).

*The latency-vs-overlap tradeoff.* Suppose backward takes 100 ms and all-reduce of a 25 MB bucket takes 10 ms.

- **Tiny buckets (1 MB):** First bucket fills very early; excellent overlap. But 25× more all-reduces, each carrying NCCL launch + ring-startup overhead. Effective bandwidth ~30% of peak. Total comm time bloats.
- **Huge buckets (200 MB):** Only 1–2 buckets total. Each is bandwidth-saturating (great per byte). But first bucket doesn't fill until very late in backward — almost no overlap. Total step time = compute + comm (sequential).
- **Medium buckets (25 MB default):** Multiple buckets fill across backward, all-reduces are bandwidth-efficient, overlap well.

The default 25 MB was tuned for typical NVLink + Infiniband. On a slower fabric, larger buckets (more bandwidth-bound regime); on very fast fabric with cheap collectives, smaller buckets give better overlap.

*The "first bucket" problem.* DDP fills buckets in **reverse forward order** so output-side layers' grads (computed first in backward) populate the first bucket. But the first bucket can't all-reduce until *all* parameters assigned to it are ready, even ones from earlier layers backward hasn't reached. If bucketing assigns a small early-layer parameter to the same bucket as the last-layer parameters, that bucket's all-reduce is delayed.

*Tunables.*

- `bucket_cap_mb` — bucket size cap. Knob you actually turn.
- `gradient_as_bucket_view` — if True, `param.grad` becomes a view into the bucket buffer, eliminating the post-all-reduce copy. Default False for backward compat; turn it on for memory and a small speedup.
- `static_graph=True` — promises the autograd graph is identical every step. Lets DDP skip per-step checks and pre-plan an optimal bucket schedule.

*Diagnosis in TraceML.* If backward time is dominated by a long tail at the end (GPU idle waiting for NCCL), you're under-overlapping — try smaller buckets. If you see many small NCCL launches and a comm-bound step, you're over-fragmenting — try larger buckets. The "right" answer is hardware-dependent.

*Mental model.* Think of bucketing like batching API calls. One request per item gives fine-grained latency but eats your rate limit and per-call overhead. Batching too aggressively means waiting to accumulate. The sweet spot is batches large enough to amortize per-call cost but small enough to pipeline. DDP buckets are exactly that pattern, applied to gradients on the network.

**Concepts introduced:** gradient bucketing rationale (latency vs bandwidth), bucket as flat byte buffer, copy-into-bucket vs `gradient_as_bucket_view`, NCCL launch overhead and ring-startup latency, reverse-order bucket fill, the "first bucket problem", `bucket_cap_mb` knob, `static_graph` optimization, batching analogy.

---

### P41: What's the difference between DDP, FSDP, and ZeRO (DeepSpeed)?

**Date:** 2026-04-24

**Short answer:** All three solve "train a model across N GPUs," but they differ in *what gets replicated vs. sharded*. **DDP** replicates the full model, gradients, and optimizer state on every rank, and only synchronizes gradients via all-reduce — simplest, fastest per step, memory-bound by single-GPU capacity. **FSDP** (PyTorch's Fully Sharded Data Parallel) shards parameters, gradients, and optimizer state across ranks; before each forward/backward of a layer it all-gathers the parameters, then frees them after — trades extra communication for the ability to fit much larger models. **ZeRO** is DeepSpeed's equivalent family of techniques, parameterized as stages 1, 2, and 3. FSDP is essentially PyTorch's native ZeRO-3.

**Long answer:**

*DDP recap.* Each rank holds a full copy of model (parameters, gradients, optimizer state). Forward and backward run independently on each rank's local minibatch. After backward, gradients are all-reduced (Q12). Optimizer.step runs locally and produces identical updated parameters. **Memory per rank ≈ model_size + grad_size + optimizer_state_size**, which for Adam is roughly 4× model size (params + grads + 2 momentum buffers) in fp32, all on every GPU.

*The memory wall.* For a 7B-parameter model in mixed precision with Adam:

- Parameters (bf16): 14 GB
- Gradients (bf16): 14 GB
- Optimizer state (Adam moments in fp32 + master fp32 weights): ~84 GB

That's ~112 GB per GPU just for state, before any activations. No single GPU has that. DDP can't run this. Sharding state across N GPUs lets each hold 1/N of it.

*FSDP — Fully Sharded Data Parallel.* Built into PyTorch (`torch.distributed.fsdp`). Core idea: at rest, each rank only holds a 1/N shard of every parameter (and grad, and optimizer state). When a layer is about to forward, FSDP issues an **all-gather** on that layer's parameter shards so every rank temporarily has the full layer's parameters in memory. Forward runs. Full parameters are freed. Same for backward: all-gather to materialize the layer, run backward, free. The gradient produced is **reduce-scattered** — each rank ends up with only its 1/N shard of the averaged gradient. The optimizer also operates only on its shard.

The collective math:

- DDP per step: 1× all-reduce of all gradients ≈ 2× model bytes communicated.
- FSDP per step: 1× all-gather params (forward) + 1× all-gather params (backward) + 1× reduce-scatter grads ≈ 3× model bytes communicated.

So FSDP costs ~1.5× more communication than DDP, in exchange for fitting a model ~N× larger. Bandwidth for memory.

FSDP wraps modules recursively; you typically wrap each transformer block as its own FSDP unit so all-gather granularity matches compute granularity. Activation checkpointing (recomputing in backward) is usually combined with FSDP for further memory savings.

*ZeRO — DeepSpeed's staged version.* DeepSpeed introduced the same idea earlier:

- **ZeRO Stage 1**: shard **optimizer state** only. Params and grads still replicated as in DDP. Saves the largest chunk (Adam moments) with minimal extra comm.
- **ZeRO Stage 2**: shard **optimizer state + gradients**. Gradients reduce-scattered instead of all-reduced.
- **ZeRO Stage 3**: shard **everything** — params, grads, optimizer state. Equivalent in spirit to FSDP.

ZeRO also adds **ZeRO-Offload** (push optimizer state to CPU RAM) and **ZeRO-Infinity** (push to NVMe). PyTorch FSDP has analogous CPU offload via `CPUOffload(offload_params=True)`.

*When to use which.*

- **DDP**: model fits comfortably on one GPU. Lowest comm overhead, simplest semantics. Right default until you hit memory wall.
- **ZeRO Stage 1 / FSDP with `SHARD_GRAD_OP`**: model + grads fit but optimizer state doesn't. Cheap upgrade from DDP.
- **FSDP full sharding / ZeRO Stage 3**: model itself doesn't fit per GPU. Pay the 1.5× comm cost.

*Orthogonal axes.* DDP/FSDP/ZeRO are all forms of **data parallelism**. Two other axes:

- **Tensor parallelism (TP)**: split a single layer's matmul across GPUs (e.g., split a `[H, 4H]` weight matrix column-wise; each GPU computes its slice; results all-reduced). Used in Megatron-LM. Communication *inside every layer's forward* — bandwidth-hungry, typically restricted to within a node where NVLink is fast.
- **Pipeline parallelism (PP)**: split layers across GPUs. Activations flow through the pipeline; micro-batches keep all GPUs busy. Adds pipeline bubble overhead but enables very deep models.

These compose: a frontier-scale training run might be DDP × TP × PP × FSDP all at once — **3D parallelism** or **4D parallelism**. Each axis has its own all-reduce / all-gather / reduce-scatter pattern.

*Mental model with RL analogy.* DDP is like having N identical agents that all do their own rollouts and average gradients at the end of each step — embarrassingly parallel except for one all-reduce. FSDP/ZeRO-3 is like the actor-learner split where the model weights themselves don't all fit in any one process; instead the system pages each layer's weights into the right GPU on demand, computes, then evicts. Pay extra communication to get a bigger effective model.

*Bottleneck profile.*

- DDP: comm time ≈ end-of-backward all-reduce. Long tail before optimizer step.
- FSDP: comm appears throughout forward and backward. Per-layer time inflated by per-layer comm.
- TP: comm is *inside* each layer.
- PP: comm is point-to-point; pipeline bubbles look like idle gaps.

A profiler that doesn't distinguish compute from comm — or aggregates across ranks — can't tell you which is biting. Per-rank visibility argument for TraceML.

**Concepts introduced:** model state breakdown, the memory wall, FSDP all-gather + reduce-scatter pattern, FSDP wrap granularity (per-layer), comm cost ratios (DDP 2× vs FSDP 3×), ZeRO stages 1/2/3 and equivalence to FSDP, ZeRO-Offload / FSDP CPU offload, tensor parallelism (Megatron-style), pipeline parallelism + micro-batches + bubbles, 3D/4D parallelism composition, activation checkpointing as orthogonal lever.

---

### P42: What happens if one rank hangs? How does NCCL detect / recover, and what's the role of `NCCL_TIMEOUT`?

**Date:** 2026-04-24

**Short answer:** All-reduce is a barrier (see [Q12](learning-qa.md#q12-what-is-nccl-all-reduce-and-why-is-it-a-barrier)) — every rank must arrive or none progress. If one rank hangs (stuck in a long forward, OOM, dead kernel, slow disk in dataloader), every other rank blocks at the next collective. NCCL has **no built-in recovery** — recovery means kill and restart. What it does have is a **watchdog thread** that monitors collective progress and aborts the process group after a configurable timeout. With `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, the abort surfaces as a Python exception with traceback rather than a silent indefinite hang. Diagnosing *which* rank is the slow one is hard precisely because the symptom shows up everywhere at once — and that's where per-rank visibility tools are essential.

**Long answer:**

*The hang scenario.* You're training on 64 GPUs across 8 nodes. At some step, one rank's dataloader stalls (slow NFS read), or one GPU thermal-throttles, or one rank hits CUDA OOM and dies, or one rank goes into a longer code path due to a data-dependent branch. Every other rank finishes its step normally and calls into the next all-reduce. The collective requires participation from all 64; 63 are waiting, 1 is missing. From those 63 ranks' perspective, the GPU is idle, the NCCL kernel is sitting in the queue waiting for its peer, Python is blocked inside the all-reduce call.

To an outside observer: GPU utilization drops to near zero on most ranks. No log lines. No exception. No progress. A bad version of this can sit there for hours.

*Why NCCL can't auto-recover.* NCCL collectives are fundamentally collective — there's no concept of "the group continues without rank 5." Once a rank has missed a collective, the communicator's state is corrupted: pending work would be wrong; ring topology has a missing node; in-flight buffers are in undefined states. The only safe response is to **abort the communicator**, propagate an error, and let the launcher restart (or skip via checkpoint-based recovery). NCCL provides primitives (`ncclCommAbort`) but doesn't decide when to use them.

*The watchdog thread.* PyTorch's NCCL backend launches a background **watchdog thread** alongside training. It periodically checks every in-flight NCCL operation: how long has it been pending? If a collective exceeds the configured **timeout**, the watchdog:

1. Logs a fat error message identifying the stuck collective and rank.
2. Calls `ncclCommAbort` on the affected communicator.
3. With `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (recommended; default in modern PyTorch), throws an exception in the main training thread on the next collective call.

Without async error handling, the abort happens but the main thread can stay stuck.

*The timeout knobs.*

- `timeout` argument to `init_process_group(..., timeout=timedelta(minutes=30))`: per-collective deadline. Historical default was 30 min — too long for interactive debugging, too short for some legitimately slow init phases.
- `NCCL_TIMEOUT` (env): NCCL-level timeout.
- `TORCH_NCCL_BLOCKING_WAIT=1`: makes collective calls block synchronously and check timeouts inline.
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`: raise exceptions on watchdog-detected timeouts.
- `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` (newer): heartbeat-based detection.
- `TORCH_NCCL_DESYNC_DEBUG=1`: when a hang is detected, log which collective on which rank diverged.
- `TORCH_NCCL_DUMP_ON_TIMEOUT=1`: dump flight recorder traces of in-flight collectives.

*What error you actually see.* With async error handling, after timeout:

```
[Rank 17] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=4128, OpType=ALLREDUCE,
  NumelIn=2621440, NumelOut=2621440,
  Timeout(ms)=1800000) ran for 1800012 ms
RuntimeError: NCCL communicator was aborted on rank 17.
```

Notice it comes from rank 17's perspective — but rank 17 might not be the slow one. Rank 17 was *waiting* for rank 5 (the actual culprit). Every rank except rank 5 will print roughly the same message at the same time. Rank 5 might print nothing (deadlocked or already crashed). This is the heart of why diagnosis is hard.

*Why hung ranks are hard to diagnose.*

- The error appears on the *witnesses*, not the culprit.
- Without per-rank instrumentation you have no record of what rank 5 was doing right before it stopped responding. OOM? Dataloader stuck on disk? Forward hitting a slower path?
- Aggregated metrics tell you nothing — the loss never got computed for that step.
- `py-spy dump --pid <rank-5-pid>` can show what Python was doing — useful but post-hoc and requires identifying the PID.
- Recent PyTorch's **flight recorder** logs every collective on every rank to a ring buffer; on a hang you can dump and diff to see which rank's sequence number diverged.

*Where TraceML fits.* The "rank visibility" pitch. If TraceML ships per-rank step-time, dataloader-time, forward-time, step-memory continuously to an aggregator (Q14, Q12), then when rank 5 starts taking 10× longer on one phase, the aggregator sees it in real time — not in a postmortem. The aggregator-with-per-rank-store architecture in `traceml.aggregator.RemoteDBStore` is exactly this: every rank's telemetry stays distinguishable. Diagnostic story: "look at TraceML's per-rank panel; find the rank whose timing went off-pattern in the last 30 seconds before the hang."

*Common causes of single-rank hangs.*

- Slow or jittery storage on one node (NFS, S3FS, hot-spot).
- One GPU thermal-throttling or hardware fault.
- OOM on one rank with dynamic memory (variable-length sequences, MoE routing).
- Data-dependent control flow — one rank takes a longer path; without `find_unused_parameters=True` (P39), DDP deadlocks.
- A bug where one rank loads a different model size or dtype, so its all-reduce expects a different number of bytes — NCCL hangs trying to match counts.
- One rank's NCCL communicator failing to establish (network glitch during `init_process_group`).

*Mental model.* All-reduce as a barrier is like a meeting where every attendee must be present before discussion starts. If one person doesn't show up, everyone else sits in the room indefinitely. The watchdog is the office manager who pokes their head in, declares the meeting cancelled after a timeout. Recovery means reschedule the meeting (restart from checkpoint). The hard part isn't the meeting failing — it's figuring out *which attendee* didn't show up. Per-rank observability is what gives you the equivalent of "I can see Bob's GPS still on the highway."

**Concepts introduced:** all-reduce as collective barrier, why NCCL has no auto-recovery, `ncclCommAbort`, NCCL watchdog thread, `TORCH_NCCL_ASYNC_ERROR_HANDLING`, `init_process_group` timeout, `TORCH_NCCL_BLOCKING_WAIT`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`, `TORCH_NCCL_DESYNC_DEBUG`, `TORCH_NCCL_DUMP_ON_TIMEOUT`, NCCL flight recorder, witness-vs-culprit diagnosis problem, common single-rank hang causes, `py-spy` for postmortem, the per-rank-visibility argument.

---

## Eager vs graph mode (`torch.compile`, JIT, fx)

### P43: What does `torch.compile` actually do? How is it different from TorchScript (the older `torch.jit.script`)?

**Date:** 2026-04-24

**Short answer:** `torch.compile` is PyTorch 2.x's graph-mode JIT: it captures your model's Python execution into an **FX graph**, lowers it through **AOT autograd** to a joint forward+backward graph, then hands that to **Inductor** which generates fused **Triton** (GPU) or **C++/OpenMP** (CPU) kernels. It's strictly a runtime accelerator — your model stays a normal Python `nn.Module`, and on a guard mismatch it falls back to eager. **TorchScript** (`torch.jit.script` / `torch.jit.trace`) is the older approach: it parses your Python source into a typed IR that runs in a separate C++ JIT runtime, primarily for **deployment** (no Python at inference time). Compile is an in-Python optimizer; TorchScript is an export format.

**Long answer:**

*The eager-mode baseline.* In eager PyTorch, every op is dispatched the moment Python hits it. `y = torch.relu(x)` walks the dispatcher (device, dtype, autograd keys), launches a CUDA kernel, returns. Each op is independent; the framework has no idea what op comes next. Great for debugging, bad for performance: you pay Python dispatch overhead per op (~5–50 µs each), can't fuse adjacent ops, can't reorder for memory locality. For a transformer block with dozens of small ops (layer norms, residuals, elementwise), launch overhead can dominate compute.

*What `torch.compile` does, end to end.* You wrap your model: `model = torch.compile(model)`. Nothing happens until first forward. On that first call:

1. **TorchDynamo** intercepts Python frame execution (PEP 523 frame evaluation API, see P44). As your forward runs, Dynamo symbolically traces the bytecode, recording every op into an **FX graph** (`torch.fx.GraphModule`). Records **guards** — runtime conditions like "input is a CUDA tensor of shape `[B, 768]`, dtype `float16`, contiguous" — that must hold for this trace to be reusable.
2. When Dynamo hits something it can't trace (data-dependent Python branch, call into a C extension it doesn't understand, a `print`), it emits a **graph break**: compiles the FX graph collected so far, runs it, then drops back to eager Python for the un-traceable bit, then starts a new trace.
3. The captured FX graph goes to **AOT autograd**, which reruns the graph through PyTorch's autograd machinery (with fake tensors) to extract the joint forward + backward graph as a single FX graph. This is what lets compile fuse ops across the forward/backward boundary.
4. The joint graph goes to a backend — by default **Inductor** — which performs op fusion, scheduling, and code generation. For GPU it emits **Triton** kernels (OpenAI's Python-embedded DSL for GPU kernel authoring). For CPU it emits C++ + OpenMP.
5. The compiled artifact is cached, keyed by guards. Next forward call: Dynamo checks guards (cheap), and if they pass, jumps directly to compiled code.

User-visible: first call is slow (compile time, often seconds), subsequent calls are faster than eager — sometimes 1.3–2× on training, more on inference.

*Why `torch.compile` is different from TorchScript.* TorchScript (PyTorch 1.x era) tried to solve a different problem: **export a model to run without Python**.

- `torch.jit.script(model)` parses Python source via an AST walker, supporting only a typed subset (no `**kwargs` magic, restricted control flow). Result: TorchScript IR you save to a `.pt` file and load in C++ (libtorch) for serving.
- `torch.jit.trace(model, example_input)` runs the model once with example inputs and records dispatched ops. Loses control flow.

TorchScript's runtime is a separate C++ interpreter / JIT. Once exported, your model is a foreign object — debugging is painful, mutation patterns restricted, many Python idioms don't work. Optimized for **deployment portability**, not training speed.

`torch.compile`, by contrast, never leaves Python. Model stays an `nn.Module`. No separate IR you save. No loss of debuggability for parts that fall back to eager. Optimizes for **training speed** (and inference within a Python process).

| Dimension              | `torch.jit.script` | `torch.compile`           |
| ---------------------- | ------------------ | ------------------------- |
| Capture mechanism      | AST parsing        | Bytecode-level frame eval |
| Coverage of Python     | Restricted subset  | Full Python (with breaks) |
| Output                 | TorchScript IR     | In-memory Triton/C++ kernels |
| Runtime                | Separate C++ JIT   | Same Python process       |
| Debuggability          | Lost               | Preserved (eager fallback) |
| Primary use case       | Deployment         | Training & inference acceleration |
| Failure on unsupported | Compilation error  | Graph break + eager fallback |

TorchScript still exists and is used for some deployment paths, but PyTorch's strategic direction is **`torch.export`** (a successor export format) plus `torch.compile` for training. `torch.jit.script` is essentially in maintenance mode.

*Failure modes specific to `torch.compile`.*

- **Excessive recompilation.** Every distinct guard set triggers a fresh compile. If input shapes vary every step, you can spend more time compiling than training. Mitigations: `torch.compile(dynamic=True)` (treats sizes as symbolic), or pad to fixed shapes.
- **Graph breaks killing the win.** A single un-traceable construct in middle of forward (e.g., `.item()` forcing GPU→CPU sync, or a `print`) splits the graph. Each half compiles, but you lose cross-fusion and add overhead at the boundary. `TORCH_LOGS="graph_breaks"` shows where they occur.
- **Eager–compiled numerical drift.** Fused kernels can change reduction order, leading to small numerical differences (1e-5 in fp16). Usually harmless; occasionally surprises someone bisecting a training loss curve.
- **Hooks bypassed.** The practically important one for instrumentation tools — see P45.

*Mental model.* Eager mode is like running an RL env step-by-step in Python, paying interpreter overhead every transition. `torch.compile` is like batch-compiling a rollout into a vectorized env that runs many steps in C — same logical computation, far less per-step overhead. TorchScript was the older approach of *exporting* the env to a different runtime entirely.

**Concepts introduced:** eager mode vs graph mode, JIT compilation, `torch.compile` pipeline, TorchDynamo (frame evaluation), FX graph, AOT autograd, Inductor backend, Triton kernels, guards & guard-keyed caching, graph break, recompilation triggers, dynamic shapes, TorchScript (`torch.jit.script` / `torch.jit.trace`), libtorch C++ runtime, `torch.export` as TorchScript's successor, deployment vs training-acceleration distinction.

---

### P44: What are TorchDynamo, AOT autograd, and `torch.fx` — how do they relate?

**Date:** 2026-04-24

**Short answer:** Three layers of PyTorch 2.x's compile stack, each owning a distinct job. **TorchDynamo** is the *frontend*: it captures your Python code into a graph by intercepting CPython's bytecode interpretation. **`torch.fx`** is the *IR* — a simple, easy-to-rewrite graph data structure that holds the captured ops. **AOT autograd** is a *middle pass*: it takes the FX forward graph, runs it through PyTorch's autograd machinery, and emits a joint forward+backward FX graph that a backend (like Inductor) can then lower to fused kernels. Pipeline: Python bytecode → Dynamo → FX graph → AOT autograd → joint FX graph → Inductor → Triton/C++.

**Long answer:**

*`torch.fx` — the IR layer.* FX is a Python-level intermediate representation. An `fx.GraphModule` is a regular `nn.Module` whose forward is generated from an `fx.Graph` — a list of nodes:

- `placeholder` (input)
- `call_function` (op like `torch.relu`)
- `call_method`
- `call_module` (a sub-`nn.Module`)
- `output`

```python
# A graph might look like:
# %x          : placeholder
# %weight     : get_attr  -> self.linear.weight
# %matmul     : call_function torch.matmul(%x, %weight)
# %relu       : call_function torch.relu(%matmul)
# return %relu
```

FX is intentionally simple — Python objects you can iterate, mutate, and re-emit as Python code. Supports **symbolic tracing** (run the model with proxy tensors that record ops) and **graph rewrites**. Existed before `torch.compile`; the foundation for quantization passes, custom transforms, early graph-mode experiments. Not a runtime — graph runs through normal PyTorch ops; FX is purely a data structure.

Why FX wasn't enough on its own: symbolic tracing trips over Python control flow that depends on tensor data. `if x.sum() > 0:` can't be traced symbolically because the proxy tensor has no real value. Workarounds were painful, low coverage. Enter Dynamo.

*TorchDynamo — the frontend.* Dynamo's trick is operating one level lower than FX: it intercepts CPython at the **bytecode** level using **PEP 523's frame evaluation API**. PEP 523 is a CPython hook (Python 3.6+) that lets you replace the function interpreting each Python frame.

Dynamo installs itself as the frame evaluator. When a frame for your compiled code is about to execute, Dynamo:

1. **Symbolically interprets the bytecode**, opcode by opcode, with abstract values for tensors (shape, dtype, device) and concrete values for Python ints, lists.
2. **Records every PyTorch op** into an FX graph being built incrementally.
3. **Records guards** — predicates over inputs that must hold for this trace to apply.
4. When it hits something un-symbolic-able (data-dependent branch, call into C code, `.item()`), emits a **graph break**: compile what it has, fall back to eager for the un-traceable region, restart tracing on the next compilable region.
5. **Rewrites the bytecode** of the frame to call the compiled FX graph (under guard check) instead of the original Python.

Structurally different from TorchScript's AST capture: Dynamo doesn't restrict your Python; it tolerates not being able to trace 100%. Result: **partial graph capture with graceful fallback**, why `torch.compile` "just works" on most existing PyTorch code.

Mental model: Dynamo is to PyTorch what a tracing JIT (LuaJIT, V8) is to general programming languages — traces hot paths into fast native code, guarded on conditions, falls back to interpretation on guard miss.

*AOT autograd — the joint-graph extractor.* By the time Dynamo finishes, you have an FX graph for **forward only**. To compile training, you need backward too — and you'd like to compile both together to fuse forward and backward ops, recompute cheaply within fused kernels.

AOT autograd:

1. Take the forward FX graph from Dynamo.
2. Run it through PyTorch's normal autograd engine, but with **fake tensors** (shape/dtype/device metadata, no real storage). Autograd builds the backward graph as a side effect.
3. Capture both graphs as a single joint FX graph: forward producing activations, backward consuming them and producing gradients.
4. **Partition** back into forward and backward subgraphs, with a clear contract for which intermediate tensors get saved (the "saved tensors" list). The partitioner can choose to *recompute* an intermediate during backward instead of saving it — trading memory for compute (activation-checkpointing-style optimizations).
5. Hand both subgraphs to the backend (Inductor) for kernel codegen.

"AOT" = ahead-of-time, in contrast to PyTorch's normal autograd which builds the backward graph **eagerly during forward**. AOT autograd does the same work, but in advance, on a captured graph, so the result is a static graph the compiler can optimize globally.

*How they fit together.*

```
Your nn.Module
     |
     | (Dynamo intercepts frame eval)
     v
+--------------------------+
| TorchDynamo              |
|  - bytecode interp       |
|  - guards                |
|  - graph breaks          |
+------+-------------------+
       |
       | FX forward graph
       v
+--------------------------+
| AOT autograd             |
|  - fake-tensor replay    |
|  - autograd → joint graph|
|  - fwd/bwd partition     |
+------+-------------------+
       |
       | FX joint graph (fwd + bwd subgraphs)
       v
+--------------------------+
| Backend (default: Inductor)
|  - op fusion & scheduling|
|  - Triton (GPU) / C++ (CPU) codegen
+------+-------------------+
       |
       | Compiled callable
       v
   Cached, guard-keyed
```

Each layer is independently usable:

- `torch.fx` for static graph rewrites without invoking Dynamo (some quantization workflows).
- Dynamo with a custom backend that's not Inductor: `torch.compile(model, backend="my_backend")` accepts any function `(GraphModule, example_inputs) -> callable`.
- AOT autograd standalone via `functorch`-derived APIs for research.

The integrated `torch.compile()` call wires all three together with sensible defaults.

*Why this layered design matters.* Pre-2.0 PyTorch tried to capture graphs at the wrong level — either too restrictive (TorchScript's typed Python subset) or too brittle (FX's symbolic tracing on raw Python). PEP 523 unlocked a new approach: capture at the bytecode level, where you can symbolically interpret without restricting source-level Python. The cost is implementation complexity (Dynamo is nontrivial code), but the win is that the average user doesn't have to think about it.

*RL analogy.* This decomposition is similar to how a modern RL framework separates: the **rollout collector** (Dynamo — captures experience), the **replay buffer / data structure** (FX — holds it), the **gradient computer** (AOT autograd — computes updates), and the **optimizer kernels** (Inductor — applies updates fast). Each is independently swappable.

**Concepts introduced:** `torch.fx` IR (placeholder/call_function/call_module/output nodes), symbolic tracing limitations, TorchDynamo, PEP 523 frame evaluation API, bytecode-level capture, guards as JIT trace-validity predicates, graph break + eager fallback, AOT autograd, fake tensors, joint forward+backward graph, fwd/bwd partitioning and recompute trade-off, Inductor as one of multiple possible backends, custom backends via `torch.compile(backend=...)`, layered compile stack, parallel to tracing-JIT-with-guards architectures (LuaJIT/V8).

---

### P45: Does `torch.compile` break module hooks / TraceML's instrumentation? If so, what's the workaround?

**Date:** 2026-04-24

**Short answer:** Yes, by default it can. When **TorchDynamo** traces past an `nn.Module` boundary, it inlines the module's forward into the captured FX graph and **bypasses `nn.Module._call_impl`** — which is exactly the entry point where PyTorch's hook firing logic lives (see [Q9](learning-qa.md#q9-what-are-hooks-and-what-does-injecting-hooks-in-process-mean) and [P48](#p48-what-is-_call_impl-and-why-does-traceml-monkey-patch-around-it-instead-of-just-using-public-hooks)). Forward/backward hooks registered on inner submodules silently stop firing inside the compiled region. Workarounds: mark instrumented modules as **opaque to Dynamo** (e.g., `torch.compiler.disable` on a function, or `torch._dynamo.disable` on a specific module), or skip `torch.compile` around the layers you're profiling. Because PyTorch's compile stack changes across releases, the exact API surface and behavior shift — this is the "PyTorch coupling" risk explicitly called out in TraceML's constraints.

**Long answer:**

*Why this happens.* PyTorch's hook firing lives in `nn.Module._call_impl`. The simplified flow:

1. Fire `forward_pre_hooks` (incl. TraceML's).
2. Run `self.forward(...)`.
3. Fire `forward_hooks` (incl. TraceML's).
4. Return result.

TraceML's `trace_model_instance(model)` walks the module tree and registers `register_forward_hook` / `register_full_backward_hook` callbacks on each submodule. In eager mode, every `submodule(x)` call goes through `_call_impl`, which fires the hooks.

Now turn on `torch.compile(model)`. Dynamo starts tracing at the outer model's frame. As it interprets bytecode, it encounters `self.layer1(x)`. Instead of stepping into `_call_impl` and recording it as a black box, Dynamo's default behavior is to **inline**: it follows the call into `_call_impl`, sees the hook-iteration logic, sees the call to `self.forward`, and continues tracing into `forward`'s bytecode. The captured FX graph ends up with the forward's *ops* directly in the parent graph — the hook iteration is *traced out*.

The compiled artifact runs only the ops Dynamo captured. Hook callbacks are not in the trace. They never fire inside the compiled region. From TraceML's perspective: the layer's forward executes, but TraceML records nothing for it.

This is silent. No exception, no warning by default. You see empty layer-time entries while the model trains fine.

*Backward hooks have a separate failure surface.* AOT autograd builds the backward graph from scratch by replaying the forward through autograd with fake tensors. The autograd engine doesn't replay user-registered backward hooks during this fake-tensor pass — they're side effects, not part of the math. Result: the backward subgraph that Inductor compiles has no backward-hook callsites.

*The mitigation toolkit.*

1. **`torch.compiler.disable`** (or older `torch._dynamo.disable`) — decorate a function so Dynamo refuses to trace into it. Dynamo emits a graph break and runs that function eagerly. Decorate the wrapper that calls `model(x)` and you opt the whole forward out of compile — hooks fire normally, but you lose the speedup. Useful as a debugging crutch.

   ```python
   @torch.compiler.disable
   def run_with_traceml(model, x):
       return model(x)  # eager, hooks fire
   ```

2. **Disable on specific submodules.** Setting `mod._dynamo_disable = True` (or using `torch._dynamo.disable(mod, recursive=True)`) tells Dynamo to treat that submodule as an opaque `call_module` node — `_call_impl` runs at execution time, hooks fire. Cost: disabled subtree is not fused with surrounding compiled graph. For TraceML's layer-level instrumentation, this is the most surgical option — apply per leaf module.

3. **`fullgraph=False` (default).** Allows graph breaks. Combined with disabled submodules, Dynamo gracefully splits the graph at the disabled boundary. With `fullgraph=True`, the same construct would error out — useful for production when you want to be sure no eager fallback is happening silently.

4. **Skip compile entirely during a profiling session.** TraceML can detect a profiling context and apply `torch._dynamo.config.disable = True` globally, or simply document: "for layer-level profiling, run without `torch.compile`; once you've identified the bottleneck, re-enable compile for the production run." Honest and reliable, less impressive demo.

5. **Inductor-side instrumentation (future).** Hook into Inductor's lowering pass to inject timing kernels around generated kernel boundaries. Catches timings *after* fusion but loses the per-`nn.Module` mapping. Closer to what `torch.profiler` does; may be where TraceML eventually integrates.

A practical TraceML pattern: when `trace_model_instance` runs, walk the module tree and apply `torch._dynamo.disable(submodule, recursive=False)` to each instrumented module. User keeps `torch.compile(model)` on the outer model; wrapped submodules become opaque `call_module` nodes; hooks fire; everything around them still compiles. May lose 10–30% of compile's win on heavily-instrumented runs.

*Detection.* `torch.compiler.is_compiling()` (PT 2.3+) returns `True` when Dynamo is currently tracing. A defensive hook:

```python
def _traceml_forward_hook(module, args, output):
    import torch
    if torch.compiler.is_compiling():
        return  # Skip; recording during trace pass would be bogus
    # ... normal recording ...
```

Won't fix the underlying bypass, but prevents bogus data from entering the database during the compile pass itself.

*Why this is the "PyTorch coupling" risk.* TraceML's value depends on hooks and patches firing reliably. Compile semantics — what Dynamo inlines vs treats opaquely, how `_call_impl` interacts with hook dispatch, what AOT autograd preserves through fake-tensor replay — change between PyTorch minor releases. PT 2.0/2.1/2.4 each shifted some defaults; PT 2.5+ added more public API around `torch.compiler.*`. The PyTorch-coupling constraint "PyTorch coupling: All auto-instrumentation depends on PyTorch internals that can change every release. Test coverage is critical" is precisely about this. The realistic engineering posture:

- Pin a tested matrix of PyTorch versions in CI; run a synthetic compiled-model regression suite (forward hooks fire? backward hooks fire? layer-time table populated?).
- On unsupported versions, emit a startup warning and fall back to coarser instrumentation.
- Document the compile-with-instrumentation story explicitly — silent missing data is the worst possible failure mode for a profiler.

*Mental model.* Compile is a tracing-JIT cache that captures and re-emits hot code, throwing away anything that doesn't look "essential" — including the hook plumbing your instrumentation library bolted onto `_call_impl`. To preserve hooks you **opt out** specific regions, telling the JIT "trust me, run this through the full Python call path." Cost: fusion you forgo across the boundary. Faster execution and complete observability of every Python-side callback are in tension; you choose per-module which one matters more.

**Concepts introduced:** `nn.Module._call_impl` as the hook dispatch site, Dynamo inlining of submodule calls, silent hook bypass under compile, AOT autograd not replaying backward hooks, `torch.compiler.disable` / `torch._dynamo.disable`, per-module opt-out via `_dynamo_disable`, `fullgraph=True` for fail-fast on graph breaks, `torch.compiler.is_compiling()` defensive check, Inductor-level instrumentation as alternative integration point, version-pinned regression matrix, the compile-vs-observability trade-off.

---

## Checkpointing & state

### P46: What is actually inside a saved checkpoint file (format, contents)?

**Date:** 2026-04-24

**Short answer:** A PyTorch checkpoint is just a **Python dictionary** that the user assembles and hands to `torch.save`. By convention it contains the model's **`state_dict`** (parameter and buffer tensors keyed by dotted names), the **optimizer `state_dict`** (per-parameter momentum/variance buffers and hyperparameters), the **scheduler state**, the current **epoch / global step**, scalar metrics, and often **RNG state** for reproducibility. Since PyTorch 1.6 the on-disk file is a **ZIP archive** containing a metadata blob plus one binary blob per tensor.

**Long answer:**

*A checkpoint is a user-constructed dict, not a magic object.* PyTorch does not define a "Checkpoint" type. Typical idiom:

```python
torch.save(
    {
        "epoch": epoch,
        "global_step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),       # AMP grad scaler
        "metrics": {"val_loss": 0.42, "val_acc": 0.91},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
        "config": cfg_as_dict,
    },
    "ckpt-step-12000.pt",
)
```

Lightning's `ModelCheckpoint`, HuggingFace's `Trainer.save_state`, custom training loops — convention layered on top of this dict + `torch.save`.

*What `model.state_dict()` is.* `OrderedDict[str, Tensor]` where keys are dotted module paths and values are **parameter tensors and persistent buffers** — *not* module objects, *not* hooks, *not* the forward graph, *not* gradients. Pure data snapshot with no live Python references to layers, so you can load into a freshly constructed model whose class lives in source code. Architecture is reconstructed from code; only numbers come from the file.

`state_dict` distinguishes **parameters** (in `nn.Parameter`, trainable) from **buffers** (in `register_buffer`, e.g., BatchNorm's running mean/variance). Both end up as plain tensors; the distinction is re-established at load time.

*What `optimizer.state_dict()` contains.*

- **`state`** — dict keyed by integer parameter id, mapping to per-param tensors. For Adam: `exp_avg` (first moment), `exp_avg_sq` (second moment), `step` (0-dim tensor or int). For SGD with momentum: `momentum_buffer`. **Same shape as the parameters they correspond to** — why optimizer state can be as big as the model.
- **`param_groups`** — the list of parameter groups with their hyperparameters and integer ids that map back into `state`. Ids are *positional* — index into the order parameters were passed to the optimizer — which is why loading requires reconstructing the optimizer with the same parameters in the same order.

If you skip optimizer state on resume, training "works" but loses the momentum/variance Adam has built up — kicks the optimizer back to cold-start, loss curve will spike. Save the optimizer.

*Scheduler, scaler, RNG.* `scheduler.state_dict()` typically contains `last_epoch` and internal counters; cosine schedules also store `base_lrs`. AMP `GradScaler.state_dict()` contains current loss scale and overflow counter. **RNG state** matters because data augmentation, dropout, some optimizers are stochastic — without restoring you won't bit-exactly reproduce a training trajectory after resume. For DDP, also capture `DistributedSampler.set_epoch` so shuffle resumes correctly.

*The on-disk format (post-PyTorch 1.6).* Open a `.pt` with `unzip -l`:

```
ckpt/
  data.pkl              <-- serialized top-level dict, with tensors replaced by storage references
  data/
    0                   <-- raw bytes of tensor 0's storage
    1                   <-- raw bytes of tensor 1's storage
    ...
  version
  byteorder
```

Pre-1.6 was a single legacy stream with tensor bytes inlined — partial loading was painful. Modern ZIP format has three advantages: (1) tensor data stored as raw contiguous binary blobs, can be **memory-mapped** at load (`torch.load(..., mmap=True)` in 2.1+), (2) inspect or extract individual tensors without deserializing the full metadata, (3) ZIP central directory means tools can read just metadata. Legacy format still readable.

Tensors stored as underlying **storage** plus stride/offset metadata — so two views into the same storage (parameter sharing) deduplicate on disk. If you accidentally checkpoint a model that hasn't had `.detach()` applied to its parameters, you can drag the autograd graph along — bloated checkpoints.

*What is *not* in a checkpoint by default.* Class definitions are not serialized — receiving Python process must already have your `model.py`. Forward graph not serialized. Hooks not serialized. CUDA stream/event state not serialized. For deployment artifact including model code, use TorchScript (`torch.jit.save`) or `torch.export`.

*Why dicts and not a custom format.* Lowest-friction container that lets every framework on top (Lightning, HF, accelerate, custom loops) agree on a convention without PyTorch dictating one. Cost: contract of what keys exist is per-codebase, why resuming someone else's checkpoint always involves `list(torch.load(path).keys())`.

**Concepts introduced:** `state_dict` as `OrderedDict[str, Tensor]`, parameters vs buffers, optimizer `state` and `param_groups`, positional parameter ids in optimizer state, scheduler state, AMP `GradScaler` state, RNG state for reproducibility, ZIP-based file format, tensor storage vs view, memory-mapped loading, deduplication of shared storages, `state_dict` is data-only, TorchScript / `torch.export` as code-included alternative.

---

### P47: What does `torch.save` / `torch.load` use under the hood, and what are the security implications around loading untrusted checkpoints?

**Date:** 2026-04-24

**Short answer:** `torch.save` and `torch.load` are thin wrappers around Python's standard serialization layer (the `pickle` module), with custom hooks (`_persistent_id` / `persistent_load`) that pull tensor storages out of the serialization stream and store them as separate binary blobs inside a ZIP archive. The serialization format is a **code-execution format**, not a data format — it contains opcodes that include "import this module," "look up this attribute," and "call this callable." Loading a `.pt` file from an untrusted source can therefore execute arbitrary code in your process. The two modern remediations are **`torch.load(..., weights_only=True)`** (PyTorch 2.0+, allowlist-based safe deserialization that became the default in 2.6) and switching to **safetensors**, a format that by construction can only contain tensors and metadata, with no executable payload.

**Long answer:**

*Under the hood: a code-capable format plus a sidechannel for tensors.* `torch.save(obj, path)` walks `obj` and serializes it. When the serializer encounters a `torch.Tensor` (or its `Storage`), PyTorch's `_persistent_id` callback intercepts it: instead of inlining bytes into the serialization stream, PyTorch returns a small token like `("storage", dtype, key, location, numel)` and writes the actual storage bytes to a separate file inside the ZIP archive (`data/<key>`). On load, `_persistent_load` sees those tokens and reconstructs `Tensor` objects by reading the corresponding blobs — optionally `mmap`-ing rather than copying. This is why checkpoints are fast to load and why you can extract individual tensors: tensor bytes never touched the serialization stream.

But the **wrapper around the tensors** — the dict structure, the `OrderedDict`, the optimizer's `param_groups`, any custom Python objects — is processed by the standard serializer. And the standard serializer is the part that bites you.

*Why the format is code-capable.* It is not a serialization of values; it is a **bytecode for a tiny stack machine** whose instructions include "import this module," "look up this attribute," and "call this callable with these arguments." Any class can define a `__reduce__` method that returns `(callable, args)`; on deserialization, the loader calls `callable(*args)` and uses the return value as the reconstructed object. This is the whole point of `__reduce__` — it lets weird objects describe how to rebuild themselves. It also lets a malicious payload say "to reconstruct me, call a system command of my choice."

A weaponized checkpoint can be created in a few lines of code: define a small class whose `__reduce__` returns `(os.system, ("malicious command",))`, place an instance into a dict, write it out. When a victim runs `torch.load("model.pt")` on the resulting file, the standard deserializer will execute that command *before* `torch.load` ever returns. No sandbox, no warning, no opt-in. The attacker now has code execution as the victim's user — read SSH keys, exfiltrate cloud credentials from `~/.aws`, install a persistent backdoor, cryptomine. On a shared cluster or CI runner with cloud roles attached, essentially game over.

This is not a PyTorch bug; it is how Python's standard serialization layer has always worked. The Python docs state at the top of that module: *"Never deserialize data received from an untrusted or unauthenticated source."* The footgun: the entire pretrained-model ecosystem (HuggingFace Hub, Civitai, random GitHub releases, Discord drops) shipped `.pt` and `.bin` files for years, treating them like data files.

*Real exploits in the wild.* Multiple security writeups have documented malicious checkpoints on public model hubs containing reverse shells, credential stealers, and crypto miners triggered on `torch.load`. HuggingFace responded by building scanners and increasingly nudging the ecosystem toward `safetensors`. The threat is not theoretical.

*Remediation 1: `torch.load(..., weights_only=True)`.* Added in PyTorch 1.13 and matured through 2.x, this flag swaps the standard deserializer for a restricted weights-only handler that only allows a small allowlist of "safe" globals: tensor types, dtypes, `OrderedDict`, common Python primitives, optimizer state shapes, etc. Anything outside the allowlist — including `os.system` or any unknown class — raises a deserialization error instead of executing. As of **PyTorch 2.6 the default flipped to `weights_only=True`**: new code that just calls `torch.load(path)` is safe by default; you must opt in to `weights_only=False` to deserialize arbitrary objects.

If you legitimately need to load a checkpoint that contains a class the safe handler doesn't recognize, you can extend the allowlist:

```python
import torch
from torch.serialization import add_safe_globals
from my_pkg import MyCustomScheduler

add_safe_globals([MyCustomScheduler])
ckpt = torch.load("ckpt.pt", weights_only=True)
```

Important: `weights_only=True` does not guard against logical attacks — if you allowlist a class with a malicious `__reduce__`, you have given it execution. The allowlist is your trust boundary; treat additions to it the way you would treat installing a new dependency.

*Remediation 2: safetensors.* `safetensors` is a format invented specifically to remove the code-capable layer from the model-weights pipeline. The file layout:

- 8-byte little-endian header length
- A JSON header listing each tensor's `dtype`, `shape`, and `(start, end)` byte offsets
- A flat blob of tensor bytes

That's the entire format. There is **no executable code path** during loading — the loader parses JSON, then `mmap`s the bytes. It cannot run `__reduce__` because there is no `__reduce__` in JSON. As a bonus it is fast (zero-copy `mmap`), language-portable (Rust loader, JS loader), and supports lazy partial loads. Cost: only tensors, not arbitrary Python objects, so optimizer state with non-tensor entries needs separate handling. HuggingFace has been migrating its hub to `.safetensors` as default, and most modern training stacks (Diffusers, Transformers, Lightning) read it natively.

*Operational guidance.*

- For **untrusted sources**: use `weights_only=True`, or insist on `safetensors`. Treat a `weights_only=False` deserialize of a network-sourced file the way you would treat `curl ... | sh`.
- For **your own checkpoints** on **your own infra**, the risk is lower but not zero — a compromised teammate or writable shared filesystem is the same attack. `weights_only=True` plus `safetensors` for model weights and a small JSON sidecar for scalar metadata is a clean default.
- Loading should ideally happen in a **subprocess** ([Q1](learning-qa.md#q1-what-is-a-subprocess)) with reduced privileges if you can't vouch for the source.
- File extensions are not formats. `.pt`, `.pth`, `.bin`, `.ckpt` are naming conventions; the actual format is determined by magic bytes.

*Why this matters for TraceML.* TraceML produces telemetry, not weights, so it isn't directly serializing checkpoints. But anything in the TraceML pipeline that cross-process-shuttles Python objects faces the same code-capability question. TraceML's choice of **msgspec** for inter-process telemetry framing is the right call: msgspec is a typed, schema-driven binary format (msgpack family) that cannot deserialize arbitrary Python callables, so a hostile training process cannot pwn the aggregator by sending a crafted telemetry frame. A quiet but important security property of the architecture.

*Mental model.* `torch.save` is "serialize the dict, with a sidechannel that pulls tensor bytes out so they can be mmap'd cheaply." `torch.load` is "deserialize that dict, with a sidechannel to load tensor bytes back in." The `weights_only` flag swaps the deserializer for a restricted one with an allowlist. `safetensors` removes the code-capable layer entirely. The unsafe default existed for historical reasons; the ecosystem is actively migrating away from it, and as of PyTorch 2.6 the safe default is finally on.

**Concepts introduced:** Python's standard serialization as a stack-machine bytecode, `__reduce__` and arbitrary callable invocation on deserialize, persistent_id / persistent_load sidechannel for tensors, `_weights_only` allowlist deserializer, `add_safe_globals` for extending the allowlist, default flip to `weights_only=True` in PyTorch 2.6, safetensors format (header + JSON + flat tensor blob), zero-copy mmap loading, threat model for model hubs, why msgspec is a safer cross-process serialization choice.

---

## PyTorch internals relevant to TraceML

### P48: What is `_call_impl`, and why does TraceML monkey-patch around it (instead of just using public hooks)?

**Date:** 2026-04-24

**Short answer:** `_call_impl` is **PyTorch's internal method on `nn.Module` that actually runs `forward()` and fires the registered hooks** — `Module.__call__` is just an alias for it (`nn.Module.__call__ = _call_impl`). TraceML monkey-patches `nn.Module.__call__` (the same dispatch path) rather than relying solely on `register_forward_*_hook` because it needs **wall-clock timing of the entire outermost forward** (including PyTorch's own hook-firing overhead) and a single guaranteed pairing of pre/post events that hooks can't deliver if a user has their own pre-hook that raises.

**Long answer:**

*Where `_call_impl` lives.* In `torch/nn/modules/module.py`, `nn.Module` defines `_call_impl`. Its body:

1. Iterate `_forward_pre_hooks` (call each, possibly mutate inputs).
2. Call `self.forward(*args, **kwargs)`.
3. Iterate `_forward_hooks` (call each with output, possibly mutate output).
4. Set up `_backward_hooks` / `_backward_pre_hooks` on the output tensor's autograd graph.
5. Return the (possibly hook-mutated) output.

PyTorch then sets `__call__: Callable[..., Any] = _call_impl`. When your code does `out = model(x)`, Python's `__call__` protocol → `nn.Module.__call__` → `_call_impl` → all of the above. **`_call_impl` and `__call__` are the same function object on the class.**

*What TraceML actually patches.* In [traceml/src/traceml/utils/patches/forward_auto_timer_patch.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/patches/forward_auto_timer_patch.py) the code captures the original method object and replaces the class-level slot:

```python
_ORIG_MODULE_CALL = nn.Module.__call__
...
def patch_forward() -> None:
    if getattr(nn.Module, "_traceml_forward_patched", False):
        return
    nn.Module.__call__ = _traceml_module_call
    nn.Module._traceml_forward_patched = True
```

So TraceML doesn't patch `_call_impl` literally; it patches `__call__`, which **is** `_call_impl` (until you replace it). The new `_traceml_module_call` wraps `_ORIG_MODULE_CALL(self, *args, **kwargs)` in a `timed_region(...)` that emits CUDA start/end events, then delegates back. Same trick for backward in [traceml/src/traceml/utils/patches/backward_auto_timer_patch.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/patches/backward_auto_timer_patch.py) (`torch.Tensor.backward` and `torch.autograd.backward`).

*Why patch instead of just using `register_forward_*_hook`?*

**1. Outer-most-only timing.** TraceML wants to time the *whole* forward pass of the user's top-level model — not every submodule. PyTorch hooks fire on every `Module` they're registered on; registering only on top misses anything that bypasses it. The patch uses thread-local depth counting:

```python
if _depth() > 0:
    return _ORIG_MODULE_CALL(self, *args, **kwargs)  # nested: skip timing
_set_depth(_depth() + 1)
try:
    with timed_region("_traceml_internal:forward_time", ...):
        return _ORIG_MODULE_CALL(self, *args, **kwargs)
```

First `model(x)` increments depth 0 → 1 and starts a CUDA event; every nested `child(x)` sees depth > 0 and just delegates. You'd have to re-implement this state machine on top of pre/post hooks, and even then you couldn't catch timing of PyTorch's *own* hook overhead.

**2. Hook-firing overhead is included.** A `register_forward_pre_hook` callback fires *inside* `_call_impl`, *after* PyTorch has already started looking up sub-hooks, copying input tuples, etc. By patching `__call__`, TraceML's timer brackets `_call_impl` from the outside — captures the full cost the user's program pays per `model(x)` invocation, including PyTorch's hook dispatch machinery.

**3. Hook re-entrance and exception safety.** If a user-registered `forward_pre_hook` raises, `_call_impl` aborts and the corresponding `forward_hook` never fires — leaving a `cuda.Event.record()` in your pre-hook unmatched. The `__call__` patch wraps in `try/finally` (`timed_region` does this in [traceml/src/traceml/utils/timing.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/timing.py)), so start/end events are always paired even if user code or PyTorch internals throw.

*The deeper "in-process" vs "public hook" tradeoff.* See [Q9](learning-qa.md#q9-what-are-hooks-and-what-does-injecting-hooks-in-process-mean) for the general split. For per-layer timing TraceML *does* use public hooks ([attach_layer_forward_time_hooks](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/layer_forward_time_hooks.py) registers `register_forward_pre_hook` + `register_forward_hook` on each leaf module). But for **whole-model forward** and **the global backward boundary**, the patch is the right tool: hooks observe individual modules; the patch observes the dispatch entry point itself.

*Why this is fragile.* `_call_impl`'s body is internal PyTorch and has changed across versions. Patching `__call__` is forward-compatible because we delegate to whatever `_ORIG_MODULE_CALL` is at import time — but the assumption that `__call__` *can* be replaced on the class is still PyTorch-coupling risk.

*Mental model.* Public hooks are *handler-level* middleware; they run after the dispatcher has decided what to do. Patching `__call__` is *gateway-level* middleware — wraps the dispatcher itself, sees the cost of dispatch, can guarantee bracketing. Cross-link to P30/P31 for how dispatch composes with the autograd dispatch key system; `_call_impl` lives one level above the dispatcher.

**Concepts introduced:** `_call_impl` (internal Module entry point), `__call__` aliasing on `nn.Module`, class-level monkey-patching vs instance hooks, hook firing order inside `_call_impl`, outer-most-only timing via thread-local depth counter, hook re-entrance / exception safety, gateway-level vs handler-level observation, PyTorch coupling risk for class slot replacement.

---

### P49: What's the exact firing order of `forward_pre_hook`, `forward_hook`, `backward_pre_hook`, `backward_hook`?

**Date:** 2026-04-24

**Short answer:** For a module `m`, `m(x)` fires hooks in this order: **`forward_pre_hook` → `forward()` → `forward_hook`** (synchronously, in registration order). When `loss.backward()` later runs, autograd traverses the graph in reverse and fires **`backward_pre_hook` → grad computation → `backward_hook`** for each module the autograd engine visits. Pre/post forward hooks fire *for every module call* in DFS order from the outside in; backward hooks fire in **reverse** order on the way out.

**Long answer:**

*Forward order, in detail.* Inside `_call_impl` (P48):

1. **Global forward pre-hooks** (`register_module_forward_pre_hook`).
2. **This module's forward pre-hooks**, in registration order. Each may return modified `inputs`.
3. **`self.forward(*args, **kwargs)`** — itself calls submodules, recursively triggering steps 1–6.
4. **This module's forward hooks**, in registration order. Each may return a modified output.
5. **Global forward hooks**.
6. Autograd-side setup: if any tensor in `output` requires grad, autograd attaches a hook node so backward hooks fire during eventual `loss.backward()`.

For `Sequential(Linear, ReLU, Linear)` followed by a loss:

```
Forward (main thread):
  Linear1.forward_pre_hook
  Linear1.forward
  Linear1.forward_hook
  ReLU.forward_pre_hook
  ReLU.forward
  ReLU.forward_hook
  Linear2.forward_pre_hook
  Linear2.forward
  Linear2.forward_hook

Backward (autograd worker thread):
  Linear2.backward_pre_hook
  Linear2 backward kernels
  Linear2.backward_hook
  ReLU.backward_pre_hook
  ReLU backward kernels
  ReLU.backward_hook
  Linear1.backward_pre_hook
  Linear1 backward kernels
  Linear1.backward_hook
```

Outermost-in for pre, innermost-out for post — DFS pre/post traversal. Each hook has access to whole-module input/output but **not** the parameter gradients of that module specifically — backward hook's `grad_input`/`grad_output` are gradients flowing through the module's input/output tensors. Parameter gradients are produced by `AccumulateGrad` nodes attached to each parameter.

This is why TraceML uses module hooks for *timing* and *activation memory* (properties of input/output flow) but reaches for parameter `.grad` directly (post-backward) for parameter gradient inspection.

*Subtleties.*

**Multi-call modules.** If `forward()` calls the *same* submodule twice (weight sharing), its pre/post hooks fire **twice per outer call**. TraceML's per-layer time hook handles this with a **per-layer FIFO deque** — pre pushes a start record, post pops. Without FIFO pairing you'd get crossed timestamps.

**The legacy `register_backward_hook` is broken.** Documented as deprecated; for some module types it fires with grad tensors that aren't actually the gradients of the module's inputs/outputs. **Always use `register_full_backward_hook` / `register_full_backward_pre_hook`** — TraceML does ([attach_layer_backward_memory_hooks](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/layer_backward_memory_hook.py) calls `register_full_backward_hook`).

*Why the order matters for TraceML's payloads.*

- **Layer forward memory** ([LayerForwardMemoryHook](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/layer_forward_memory_hook.py)) registers a `forward_hook` (post-only): inspects the output tensor and sums `numel * element_size`. Pre-hook can't see output yet.
- **Layer forward time** ([layer_forward_time_hooks.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/layer_forward_time_hooks.py)) registers both pre and post: pre records `cpu_start` + acquires CUDA `start_event`; post records `cpu_end` + acquires `end_event`. The interval bounds per-call layer time. CUDA events queued asynchronously on the current stream ([Q15](learning-qa.md#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread)) — pair correctly bounds GPU work without forcing host sync.
- **Layer backward memory** ([LayerBackwardModuleHook](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/layer_backward_memory_hook.py)) registers `register_full_backward_hook` (post-only): sums bytes of `grad_output`.
- **Model-level peak memory** ([model_forward_memory_hook.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/model_forward_memory_hook.py)) uses both: pre calls `torch.cuda.reset_peak_memory_stats()`, post reads `torch.cuda.max_memory_allocated()`. Order is essential.

*The autograd-graph subtlety.* Forward hooks fire *during* forward. Backward hooks are *registered* during forward (PyTorch wires them into the autograd graph as `grad_fn` callbacks) but *fire* later, when `loss.backward()` runs. This is why TraceML's [trace_step](https://github.com/Pendu/traceml/blob/main/src/traceml/instrumentation.py) wraps both: `forward_auto_timer()` + `backward_auto_timer()` are both active inside the `with` block, and the step boundary's `flush_step_events(model, step)` collects both forward and backward events into the same `StepTimeBatch`.

*Mental model.* Forward hooks = breakpoints set on function entry/exit of each layer. Backward hooks = breakpoints inserted into a tape that the autograd engine plays in reverse. They share an interface but live in different execution contexts: forward is direct call stack, backward is autograd's worker traversing a DAG. "In registration order" applies to multiple hooks on the *same* module; "DFS pre/post" applies to ordering *across* nested modules.

**Concepts introduced:** module hook firing order (forward outside-in pre, inside-out post; backward reverse), `_call_impl` execution sequence, global vs per-module hooks, `register_full_backward_hook` vs deprecated `register_backward_hook`, FIFO pairing for shared/re-entered modules, autograd graph as deferred hook registration, why TraceML's per-layer memory uses post-hooks but per-layer time uses pre+post pairs.

---

### P50: How does PyTorch's built-in profiler (`torch.profiler`) differ from TraceML's approach? Where does TraceML do better, and where does the profiler do better?

**Date:** 2026-04-24

**Short answer:** **`torch.profiler` is a *trace-record-and-export* tool**: you wrap a few iterations in a profiler context, it instruments the dispatcher and CUDA via Kineto/CUPTI, dumps a Chrome-tracing JSON or TensorBoard file, and you analyze offline. **TraceML is a *continuous, low-overhead live monitor*** designed to run for an entire training job and surface bottlenecks in real time on a terminal/dashboard, with framework adapters (HF, Lightning) that need zero user code changes. Profiler wins on per-op detail and kernel-level traces; TraceML wins on always-on operation, per-rank live aggregation, and zero user instrumentation.

**Long answer:**

*What `torch.profiler` does.* The modern profiler hooks into:

- **The PyTorch dispatcher** ([P30/P31](#p30-how-does-y-torchrelux-get-from-python-all-the-way-to-a-cuda-kernel-the-dispatcher-path)) via the **RecordFunction** mechanism — every dispatched op (`aten::mm`, `aten::add`) emits a record event with start/end times.
- **CUPTI** (CUDA Profiling Tools Interface) via **Kineto**, NVIDIA's library for collecting CUDA driver/runtime events: kernel launches, memcpy operations, stream sync points.
- **Custom user ranges** via `with profiler.record_function("my_op"):`.

Output: a `prof` object you can print as a table, export as Chrome trace (`prof.export_chrome_trace("trace.json")`), or export to TensorBoard.

The recommended usage is the **schedule API**: warm up for N steps, record M steps, repeat — because *recording every op for an entire training job is too expensive*. CUPTI synchronizes streams and adds per-kernel overhead measured in microseconds, which adds up for models doing 10k ops per step.

*What TraceML does, by contrast.* From the architecture (and grounded in [trace_step](https://github.com/Pendu/traceml/blob/main/src/traceml/instrumentation.py) + [forward_auto_timer_patch](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/patches/forward_auto_timer_patch.py)):

- Wraps the **outermost forward and backward** with CUDA events from a [reusable event pool](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/cuda_event_pool.py) — one start + one end per phase, not one per kernel.
- Per-layer hooks attach only to leaf modules and only when running in the `deep` profile.
- Best-effort try/except + queues with maxsize ([flush_layer_forward_time_buffers](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/layer_forward_time_hooks.py) silently drops if queue is full) so backpressure can never block training.
- **Aggregator runs out-of-process** (separate Python interpreter, TCP-connected) — zero impact on training process's GIL or memory.
- Renders to Rich (terminal) or NiceGUI (web dashboard) in real time, refreshing once per second by default (`TRACEML_INTERVAL=1.0`).

*Where `torch.profiler` is strictly better.*

| Feature | torch.profiler | TraceML |
| --- | --- | --- |
| Per-op kernel attribution | Yes (RecordFunction + CUPTI) | No (phase + leaf-module level only) |
| Chrome/Perfetto trace timeline | Yes | No |
| Memcpy H2D/D2H individual events | Yes | No (only aggregate via process_sampler) |
| CUDA kernel duration breakdown | Yes (per-kernel) | Aggregate per phase |
| Memory allocator timeline | Yes (`profile_memory=True`) | Peak only (max_memory_allocated) |
| Stack traces per op | Yes (`with_stack=True`) | No |
| Multi-GPU NCCL collective timing | Yes (CUPTI sees collective kernels) | Not natively |

If you've narrowed a bug to "Linear layer N is slow" and want to know *which kernel inside it*, `torch.profiler` is the right tool.

*Where TraceML wins.*

**1. Continuous-during-run with bounded overhead.** `torch.profiler` is designed for short windows. TraceML records O(1) events per phase regardless of model depth. The CUDA event pool ([cuda_event_pool.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/cuda_event_pool.py), pre-allocated 2000 events) means even per-layer instrumentation amortizes to acquire/release on a deque. Leave on for an 8-hour training run.

**2. Live, glanceable signal.** `torch.profiler` produces an artifact you analyze later. TraceML shows a Rich table that updates every second: step time mean/p95/p99, per-rank GPU utilization, dataloader vs forward vs backward vs optimizer breakdown. For "my training is slower than yesterday," that loop is seconds vs minutes.

**3. Zero-code framework integration.** `traceml run train.py` works without touching `train.py`. The CLI sets env vars, the executor uses `runpy.run_path` ([executor.py](https://github.com/Pendu/traceml/blob/main/src/traceml/runtime/executor.py)), patches install themselves. `torch.profiler` requires you to add `with profile(...)` to your code. For HF Trainer / Lightning users, TraceML provides drop-in callbacks ([traceml/src/traceml/integrations/](https://github.com/Pendu/traceml/blob/main/src/traceml/integrations/)).

**4. Per-rank, multi-process aggregation.** In DDP, each rank runs its own TraceML in-process agent, all shipping over TCP to a single aggregator that interleaves them in one display. `torch.profiler` produces one trace file per rank to manually combine; doesn't show "rank 3 is the straggler" live.

**5. Fail-open semantics.** `torch.profiler` errors typically fail your training run. TraceML's hooks all wrap in `try/except` and print to stderr — instrumentation can break and your job continues.

*Where they're equivalent.* Both ultimately rely on `torch.cuda.Event(enable_timing=True)` for GPU timing — there's no other accurate way to measure GPU duration without a synchronize. Both give wall-clock + GPU time decomposition. Both can attribute to user-named regions.

*The complementary stack.* These tools are not competitors. Realistic flow:

1. Run training under TraceML. Notice "step time p95 has doubled" or "backward time on rank 2 is 4x rank 0."
2. Suspect a specific layer or NCCL issue.
3. Reproduce with `torch.profiler.profile(...)` for 5 steps to get per-op trace.
4. Open in Perfetto, find the offending kernel.

Standard "live monitor + on-demand deep tool" split.

*Mental model.* `torch.profiler` is **strace + perf** for PyTorch — incredibly detailed, expensive enough that you only run it on demand. TraceML is **htop + Datadog** for PyTorch training — lightweight, always-on, surfaces the right level of detail to spot anomalies, then you pull out the heavy tool for diagnosis.

**Concepts introduced:** `torch.profiler` as RecordFunction + Kineto + CUPTI integration, dispatcher-level op recording, Chrome trace / Perfetto / TensorBoard handlers, profiler scheduling (warmup/active/repeat), CUPTI overhead and stream sync requirements, continuous-vs-windowed observability, fail-open instrumentation, observability vs profiling distinction, complementary tooling stack pattern.

---

### P51: Which `torch.cuda.*` APIs does TraceML rely on, and how stable are they across PyTorch versions?

**Date:** 2026-04-24

**Short answer:** TraceML touches a small, mostly **stable public surface** of `torch.cuda`: `Event(enable_timing=True)`, `Event.record/.query/.elapsed_time`, `current_device`, `set_device`, `is_available`, `memory_allocated`, `max_memory_allocated`, `memory_reserved`, `max_memory_reserved`, `reset_peak_memory_stats`, `device_count`, `get_device_properties`. These have been stable for years. The risk surface is narrower: `nn.Module.__call__` patching ([P48](#p48-what-is-_call_impl-and-why-does-traceml-monkey-patch-around-it-instead-of-just-using-public-hooks)) and `torch.Tensor.backward` / `torch.autograd.backward` patching, which depend on PyTorch's *internal dispatch shape*, not on `torch.cuda` per se.

**Long answer:**

*The actual `torch.cuda.*` call sites* (from `grep` over the source):

**Event timing — foundation of GPU duration measurement.**
- `torch.cuda.Event(enable_timing=True)` — [cuda_event_pool.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/cuda_event_pool.py). Allocates a CUDA event capable of timing.
- `event.record()` — implicitly recorded on the current stream by `timed_region` and the layer hooks.
- `event.query()` — non-blocking check whether all preceding stream work has completed. Returns bool. **Critical for "no synchronization" promise** — TraceML never blocks the host waiting on the GPU.
- `event.elapsed_time(other_event)` — milliseconds between two events on the same stream.

**Device management.**
- `torch.cuda.is_available()` — returns False if no CUDA driver/devices.
- `torch.cuda.current_device()` — int device index for the calling thread.
- `torch.cuda.set_device(idx)` — sets the CUDA context's current device for this thread.
- `torch.cuda.device(idx)` (context manager) — temporarily switches device.
- `torch.cuda.device_count()` — for DDP rank → device mapping.
- `torch.cuda.get_device_properties(i).total_memory` — static device VRAM total.

**Memory accounting (caching allocator stats).**
- `torch.cuda.memory_allocated(device)` — live-allocated bytes.
- `torch.cuda.max_memory_allocated(device)` — peak since last reset.
- `torch.cuda.memory_reserved(device)` — bytes the caching allocator has grabbed from the driver.
- `torch.cuda.max_memory_reserved(device)` — peak reserved since last reset.
- `torch.cuda.reset_peak_memory_stats(device)` — resets the peak watermarks.

That's the entire `torch.cuda.*` dependency surface. ~12 functions, all in the public API.

*Stability assessment.*

**High stability** (these have been stable since ~PyTorch 1.0):
- `is_available`, `current_device`, `set_device`, `device_count`, `get_device_properties` — fundamental device API.
- `Event(enable_timing=True)`, `record()`, `query()`, `elapsed_time()` — direct mirror of CUDA driver `cudaEventCreate / Record / Query / ElapsedTime`. Stable as long as CUDA itself is stable.
- `memory_allocated`, `memory_reserved`, `max_*`, `reset_peak_memory_stats` — caching allocator APIs, stable since ~PyTorch 1.4. Meaning shifted slightly when expandable-segments allocator landed (PyTorch 2.1's `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), but API contract unchanged.

**Medium-stability concerns.**
- The **per-thread default stream** semantics ([Q15](learning-qa.md#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread)) affect *what stream `event.record()` uses by default*. TraceML doesn't use `set_stream` explicitly, so it inherits whatever stream the user's code is running on — correct for measuring user work.

**Low-stability concerns (the real coupling risk).** These are *not* `torch.cuda.*` but they're what TraceML's "PyTorch coupling" constraint warns about:

- **`nn.Module.__call__` slot replacement** ([forward_auto_timer_patch.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/patches/forward_auto_timer_patch.py)). PyTorch reserves the right to make `_call_impl` a C++ method or move dispatch into TorchScript. The day they do, `nn.Module.__call__ = my_func` either fails silently or breaks. **Highest-risk patch in the codebase.**
- **`torch.Tensor.backward` / `torch.autograd.backward` replacement** ([backward_auto_timer_patch.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/patches/backward_auto_timer_patch.py)). Same risk class.
- **`register_full_backward_hook`** ([layer_backward_memory_hook.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/layer_backward_memory_hook.py)). Public API, but the contract for `grad_input` / `grad_output` shapes when the module's forward returns non-Tensor outputs has been quietly tightened across versions.
- **`module._traceml_*` attribute attachment** on user models. Nothing in PyTorch promises `nn.Module` will tolerate arbitrary attributes; if PyTorch ever uses `__slots__` on Module (unlikely but possible), this breaks.

*Why TraceML emphasizes test coverage.* "All auto-instrumentation depends on PyTorch internals that can change every release. Test coverage is critical." TraceML's CI should pin and test against multiple PyTorch versions (2.5, 2.6, latest nightly). Specifically, the smoke tests should verify:

1. `nn.Module.__call__` can still be reassigned at the class level and the new function is invoked when the user calls `model(x)`.
2. `torch.Tensor.backward` reassignment still intercepts user `loss.backward()` calls.
3. `torch.cuda.Event(enable_timing=True).elapsed_time(other)` still returns a float in milliseconds.
4. `torch.cuda.max_memory_allocated()` returns a peak that resets via `reset_peak_memory_stats`.

If any of those fail on a new PyTorch version, the patches need updating *before* shipping support for that version. The reliability boundary that makes a "lightweight, real-time bottleneck finder" trustworthy.

*Mental model.* TraceML's `torch.cuda.*` use is like depending on POSIX file APIs — boringly stable, well-specified, unlikely to break. TraceML's `nn.Module.__call__` and `torch.Tensor.backward` patches are like depending on undocumented kernel symbols — works today, may not next release, requires per-release smoke tests. Risk concentration is in monkey-patches, not CUDA APIs.

**Concepts introduced:** `torch.cuda.Event` semantics (record/query/elapsed_time), per-thread default stream inheritance, caching allocator stats (allocated vs reserved, current vs peak), expandable-segments allocator, public-API stability vs internal-symbol stability, monkey-patch as the real coupling risk vector, version-pinning and smoke-test discipline for PyTorch-coupled libraries.

---

### P52: How does TraceML measure per-layer memory, and what's the relationship to `torch.cuda.memory_allocated`?

**Date:** 2026-04-24

**Short answer:** TraceML measures per-layer memory **structurally, by summing the byte size of tensors observed at module boundaries** — *not* by deltas of `torch.cuda.memory_allocated`. Forward hooks read `output.numel() * output.element_size()` for activation memory; backward hooks read the same for `grad_output` for gradient memory. This avoids the caching-allocator distortion problem entirely (see [P33](#p33-how-does-pytorch-report-gpu-memory-memory_allocated-max_memory_allocated-the-caching-allocator) / [P34](#p34-why-is-reported-memory-sometimes-lower-than-nvidia-smi-shows-caching-allocator-fragmentation)) — but it also means TraceML measures *logical tensor bytes*, not *physical bytes the allocator actually held*. The whole-step `torch.cuda.max_memory_allocated` is captured separately at the step boundary.

**Long answer:**

*Why the obvious approach (`memory_allocated` deltas around each layer) doesn't work.* The naive idea: measure `memory_allocated()` before the layer's forward, after, take the difference. The problem is the **caching allocator** (P33/P34):

- PyTorch never returns memory to the CUDA driver during a training step — caches freed blocks for reuse.
- When layer N's forward produces an activation reused by layer N+1's backward, the allocator may *reuse* a block freed earlier in the step, so `memory_allocated` after N's forward might be *lower* than before (negative delta!) or unchanged even though a new tensor was created.
- For activation checkpointing or in-place ops, the same physical block holds different tensors at different points — `memory_allocated` doesn't reflect "this layer's contribution."

The delta approach gives noise dominated by allocator caching policy, not by your model's structure.

*What TraceML actually does.* The forward memory hook in [layer_forward_memory_hook.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/layer_forward_memory_hook.py):

```python
def __call__(self, module: nn.Module, inputs: Any, output: Any):
    total_bytes = 0.0

    def accumulate(obj: Any) -> None:
        nonlocal total_bytes
        if isinstance(obj, torch.Tensor):
            total_bytes += float(obj.numel() * obj.element_size())

    if isinstance(output, torch.Tensor):
        accumulate(output)
    elif isinstance(output, (list, tuple)):
        for o in output:
            accumulate(o)
    elif isinstance(output, dict):
        for o in output.values():
            accumulate(o)

    if total_bytes > 0:
        _layer_forward_memory_buffer.setdefault(self.model_id, []).append(
            (self.layer_name, total_bytes)
        )
```

`numel * element_size` over output tensors — the **logical activation size**. For `Linear(1024, 4096)` on batch 32 fp32: `output.shape = (32, 4096)`, `output.numel() = 131072`, `element_size() = 4`, so 524 KB. That's the actual size of the activation tensor regardless of what the allocator did.

Backward gradient memory ([layer_backward_memory_hook.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/layer_backward_memory_hook.py)) uses the same approach on `grad_output`.

*Aggregation across multiple invocations.* From [layer_memory_common.py](https://github.com/Pendu/traceml/blob/main/src/traceml/samplers/layer_memory_common.py):

```python
def aggregate_layer_memory_payload_max(layers):
    """
    Memory is a capacity metric, so repeated observations within a step do not
    add together. We conservatively track the maximum observed value per layer.
    """
    agg = {}
    for layer_name, mem in layers:
        prev = agg.get(layer_name)
        agg[layer_name] = mem if prev is None else max(prev, mem)
```

If a shared module is called K times in one forward (tied embeddings), TraceML reports `max` of the K activation sizes, not sum. Memory is a *capacity* metric, not a *throughput* metric.

*What `torch.cuda.memory_allocated` IS used for.* At the **whole-step level**:

- [StepMemoryTracker.reset](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/step_memory.py) calls `torch.cuda.reset_peak_memory_stats(device)` at step start.
- `StepMemoryTracker.record` calls `torch.cuda.max_memory_allocated(device)` and `torch.cuda.max_memory_reserved(device)` at step end.

This gives "peak memory footprint of one whole training step" — right granularity for `memory_allocated`-based measurement, because at step boundaries the allocator state is well-defined. Same pattern one level finer in [model_forward_memory_hook.py](https://github.com/Pendu/traceml/blob/main/src/traceml/utils/hooks/model_forward_memory_hook.py): pre-hook resets peak stats, post-hook reads max_allocated. "Peak memory during the model's forward."

*The three layers of memory observation.*

| Granularity | Method | Source | Semantics |
| --- | --- | --- | --- |
| Per-layer activation | `tensor.numel() * element_size()` on hook output | `layer_forward_memory_hook` | Logical bytes per activation tensor |
| Per-layer gradient | Same on `grad_output` | `layer_backward_memory_hook` | Logical bytes per gradient tensor |
| Whole forward peak | `max_memory_allocated` between pre/post model hooks | `model_forward_memory_hook` | Real allocator peak during forward |
| Whole step peak | `max_memory_allocated` between step start/end | `StepMemoryTracker` | Real allocator peak across forward+backward+optimizer |

These don't add up consistently because they measure different things. Sum of per-layer activations ≠ whole-forward peak (allocator can reuse freed blocks; some activations released before others appear). Whole-forward peak ≤ whole-step peak.

*Relationship to `memory_reserved`.* Reserved = what the caching allocator has grabbed from the driver — what `nvidia-smi` reports. Allocated ≤ Reserved. The gap is the allocator's free pool. For diagnosing OOMs, **reserved matters**. For attributing memory to *your model*, **allocated matters**. TraceML reports both at step level.

*Why the structural approach (numel × element_size) is a feature.* Per-layer attribution should be **causal** — "this is how much memory layer N contributes." The allocator-delta approach is **correlational with allocator policy**, the opposite of what you want. By measuring tensor sizes directly, TraceML answers "if I removed this layer's activations from the autograd graph (e.g., by activation checkpointing), how much would I save in the *worst* case?" That's the actionable number.

The tradeoff: TraceML *cannot* tell you "this layer caused fragmentation" or "this layer triggered a `cudaMalloc`." For those, use `torch.profiler` with `profile_memory=True` to get the per-allocation timeline, or PyTorch's memory snapshot tool (`torch.cuda.memory._record_memory_history()`).

*Worked example.* `Linear(1024, 4096)` followed by `ReLU()` on batch 32, fp32:
- Linear forward output: 32 × 4096 × 4 = 524,288 bytes.
- ReLU forward output: same shape, same 524,288 bytes.
- Sum of per-layer activations TraceML reports: 1,048,576 bytes (~1 MB).
- But ReLU is in-place if `inplace=True` — same physical memory.
- `memory_allocated` delta would show only ~524 KB grew (ReLU reused).
- TraceML still reports both layers at 524 KB each because it's measuring *the tensors that exist*, even if they alias.

For "where is my activation memory going?", TraceML's number is correct. For "why is my GPU memory at 8 GB?", the step-level `max_memory_allocated/reserved` is correct.

*Cross-references.*

- [Q9](learning-qa.md#q9-what-are-hooks-and-what-does-injecting-hooks-in-process-mean) — why hooks are the right vehicle for in-process per-layer attribution.
- [Q15](learning-qa.md#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread) — GPU memory state separate from anything host can directly observe.
- [P33](#p33-how-does-pytorch-report-gpu-memory-memory_allocated-max_memory_allocated-the-caching-allocator) / [P34](#p34-why-is-reported-memory-sometimes-lower-than-nvidia-smi-shows-caching-allocator-fragmentation) — caching allocator foundation.
- [P49](#p49-whats-the-exact-firing-order-of-forward_pre_hook-forward_hook-backward_pre_hook-backward_hook) — why post-hook is the right place to read output sizes.

*Mental model.* Per-layer memory is **a structural property of the network** (how many activation bytes does layer N's output occupy). Step-level memory is **a property of the allocator's behavior under your workload** (what watermark did your allocator hit). Mixing the two leads to "but the deltas don't add up!" confusion — they're answering different questions, which is why TraceML maintains both granularities side by side.

**Concepts introduced:** structural vs delta-based memory measurement, why caching-allocator deltas mislead per-layer attribution, `numel × element_size` as logical tensor footprint, max-aggregation rationale for repeated-call modules (capacity not throughput), the three granularities (per-layer / whole-forward / whole-step), allocated vs reserved distinction, in-place op handling under structural measurement, when to fall back to `torch.profiler` `profile_memory=True` for true allocation timeline.

---


<!-- New entries appended below. Format matches the existing style above. New questions get the next P-number; pick the right sub-topic section and add a TOC entry at the top. -->
