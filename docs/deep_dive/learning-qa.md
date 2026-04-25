# TraceML Architecture — Learning Q&A

Personal Q&A log for deepening understanding of the TraceML architecture.
Questions from Abhijeet, answers from Claude. No question is too basic.

**Started:** 2026-04-24
**Reference docs:** `traceml/docs/developer_guide/architecture.md`

---

## How this is organized

Questions are grouped by topic into the sections below. Each section accumulates new questions over time. **Question numbers are global** (Q1, Q2, …) and don't reset per category — that way concept cross-references like "see Q5 on the kernel" stay stable as the file grows. New questions get the next global number and slot into the right section.

Categories so far (more can be added as topics emerge):

- **OS Basics** — kernel, processes, signals, file descriptors, pipes, fork/exec
- **Networking** — TCP, ports, sockets, RDMA, Infiniband
- **Python Internals** — bytecode, GIL, hooks/monkey-patching, multiprocessing
- **CUDA & GPU** — context, streams, events, GPU memory model
- **Distributed Training** — rank, NCCL, all-reduce, DDP, collectives
- **TraceML Architecture** — server-side/user-side, long-running, design rationale

**PyTorch fundamentals & internals** (tensors, nn.Module, autograd, optimizers, DataLoader, dispatch, AMP, DDP/FSDP, compile, TraceML-relevant internals) live in their own companion document: [traceml_pytorch_qa.md](traceml_pytorch_qa.md). That file uses **P1, P2, …** numbering. Cross-references between files use clickable Markdown links.

**Code walkthroughs** (file-by-file readings of the TraceML codebase) live in another companion: [traceml_learning_code_walkthroughs.md](traceml_learning_code_walkthroughs.md). That file uses **W1, W2, …** numbering. So the three files have independent number sequences (Q for concepts, P for PyTorch, W for walkthroughs) that never collide.

---

## Table of Contents

### OS Basics

- [Q1: What is a subprocess?](#q1-what-is-a-subprocess)
- [Q5: OS fundamentals — kernel, process internals, pipes, sockets](#q5-os-fundamentals--kernel-process-internals-pipes-sockets)
- [Q7: Spawning — fork, exec, and multiprocessing start methods](#q7-spawning--fork-exec-and-multiprocessing-start-methods)

### Networking

- [Q10: What is TCP concretely, and what's a port?](#q10-what-is-tcp-concretely-and-whats-a-port)
- [Q14: What is RDMA / Infiniband and why does it matter for multi-node training?](#q14-what-is-rdma--infiniband-and-why-does-it-matter-for-multi-node-training)

### Python Internals

- [Q6: Python internals — bytecode, GIL, CPU-bound vs I/O-bound](#q6-python-internals--bytecode-gil-cpu-bound-vs-io-bound)
- [Q9: What are hooks, and what does "injecting hooks in-process" mean?](#q9-what-are-hooks-and-what-does-injecting-hooks-in-process-mean)

### CUDA & GPU

- [Q11: What is a CUDA context, and why is it fork-unsafe?](#q11-what-is-a-cuda-context-and-why-is-it-fork-unsafe)
- [Q15: What is a CUDA stream and how does it differ from a CPU thread?](#q15-what-is-a-cuda-stream-and-how-does-it-differ-from-a-cpu-thread)

### Distributed Training

- [Q4: What is a GPU rank?](#q4-what-is-a-gpu-rank)
- [Q12: What is NCCL all-reduce, and why is it a barrier?](#q12-what-is-nccl-all-reduce-and-why-is-it-a-barrier)

### PyTorch (separate file)

See [traceml_pytorch_qa.md](traceml_pytorch_qa.md) — has P1–P52 covering tensor fundamentals, nn.Module mechanics, autograd, optimizers, DataLoader, CUDA dispatch path & memory, mixed precision (AMP), distributed (DDP/FSDP), eager vs graph mode (torch.compile), checkpointing, and PyTorch internals relevant to TraceML.

### TraceML Architecture

- [Q2: What is "server-side" vs "user-side"?](#q2-what-is-server-side-vs-user-side)
- [Q3: What does "long-running" mean?](#q3-what-does-long-running-mean)
- [Q8: Why TraceML uses subprocesses (expanded)](#q8-why-traceml-uses-subprocesses-expanded)

### Code Walkthroughs (separate file)

See [traceml_learning_code_walkthroughs.md](traceml_learning_code_walkthroughs.md) — currently has **W1: cli.py — top-level launcher and process orchestrator**.

---

## OS Basics

### Q1: What is a subprocess?

**Date:** 2026-04-24
**Source:** `architecture.md` diagram (the "subprocess" label on the arrows from `TraceOpt CLI`)

**Short answer:** A subprocess is a separate OS process launched from another process. In Python, you spawn one with `subprocess.Popen(...)`. The parent and child have **separate memory, separate Python interpreters, and separate file descriptors** — they communicate only through channels the OS allows: inherited environment variables (set at spawn time), pipes (stdin/stdout/stderr), or network sockets. In TraceML, the CLI spawns two subprocess trees: (1) the **aggregator** (one Python process), (2) **torchrun** (one process, which itself spawns N training workers — so the CLI is the grandparent of the training ranks).

**Long answer:**

*Process vs thread — the foundational distinction.*

- A **thread** is a scheduling unit inside a process. Threads in the same process share memory: one thread can read/write variables owned by another.
- A **process** is a full isolation boundary: its own virtual address space (enforced by the OS via page tables), own open-file table, own signal handlers. One process cannot directly read another's memory.
- A **subprocess** is just a process spawned by another process — the "sub-" prefix names the parent-child relationship, nothing more.

*Why this matters in Python specifically.*
Python has the **GIL** (Global Interpreter Lock): within one Python process, only one thread can execute Python bytecode at a time. Threads in Python are good for I/O-bound work (waiting on network/disk) but not for CPU-bound Python parallelism. If you want true CPU parallelism — e.g., N training workers each running a forward pass concurrently — you need **processes**, not threads. This is the core reason PyTorch distributed training uses one process per GPU.

*How spawning actually works on Linux.*

- `fork`: the kernel duplicates the current process. Memory is **copy-on-write** — pages are shared until one side writes, at which point the OS makes a private copy. Fast but inherits all of Python's loaded state.
- `execve` (the OS system call family commonly called "exec"): the kernel replaces the current process's code with a new program (same PID, fresh code). After it returns, none of the pre-fork Python state remains.
- `subprocess.Popen` is a wrapper over fork + exec — you get a clean new program connected to the parent only by env vars and stdio pipes.

*Why TraceML uses subprocesses (not threads).*

- **Crash isolation**: if the aggregator segfaults, your training keeps running. Impossible if it were a thread in the same process.
- **GIL avoidance**: aggregator work (TCP I/O, SQL, Rich rendering) doesn't compete with training code for the GIL.
- **Symmetry of ranks**: with 4 GPUs, all 4 training processes are peers pushing to one external aggregator — no "special" rank hosting the aggregator thread.
- **torchrun convention**: PyTorch's distributed launcher is itself a subprocess-spawning tool, so composing with it means the CLI is naturally a subprocess orchestrator.

*Concrete code.* In [traceml/src/traceml/cli.py](traceml/src/traceml/cli.py), search for `subprocess.Popen` — you'll find two spawn sites (aggregator, torchrun). Note `start_new_session=True`: this puts the child in a new process group, so a single `os.killpg` on Ctrl-C cleanly tears down the whole tree.

**Related files:** [traceml/src/traceml/cli.py:397](traceml/src/traceml/cli.py#L397), [traceml/src/traceml/cli.py:424](traceml/src/traceml/cli.py#L424)
**Concepts introduced:** process vs thread, GIL, fork + exec, copy-on-write memory, process group, signal propagation, `subprocess.Popen`

---

### Q5: OS fundamentals — kernel, process internals, pipes, sockets

**Date:** 2026-04-24
**Source:** clarifications on Q1 ("own open-file table, own signal handlers", "pipes (stdin/stdout/stderr)", "network sockets", "kernel")

**Short answer:** The **kernel** is the core of the operating system — the privileged code that manages hardware and enforces isolation between programs. A **process** is a running program whose state the kernel tracks: its own address space, open-file table, and signal handlers. **Pipes** are in-memory unidirectional byte channels between processes, used for stdin/stdout/stderr. **Sockets** are bidirectional network-style channels that also work on localhost.

**Long answer:**

*The kernel.* Computers run two kinds of code: **kernel mode** and **user mode**. The kernel is the code that runs in privileged mode (Ring 0 on x86) — it can talk to hardware directly, allocate physical memory, switch which process is running on a CPU core. Your Python program runs in unprivileged mode. When it wants to do something that requires privilege (open a file, allocate memory, send a packet), it makes a **syscall** — a controlled entry into the kernel. The syscall traps into kernel mode, the kernel does the work, returns. In Linux, the kernel is a single ~28-million-line C program. Python's `os`, `subprocess`, `socket` modules are thin wrappers over syscalls. Every time `subprocess.Popen` runs, it makes syscalls. Every time TCP data moves, syscalls. Python orchestrates; the kernel does the heavy lifting.

*Process internals in detail.* A process is a kernel-managed data structure (`task_struct` in Linux). Its per-process state:

- **Address space** — a map from virtual addresses (what pointers effectively look like) to physical memory, enforced by the hardware MMU via **page tables**. If process A tries to read an address not mapped for A, the MMU faults → segfault. Each process thinks it has the full 2^64-byte address space. Different page tables are why processes can't read each other's memory.
- **Open-file table** — an array indexed by **file descriptor** (fd). fd 0 = stdin, fd 1 = stdout, fd 2 = stderr. Opening a file returns a new fd (typically 3+). Sockets, pipes, and regular files all share the same fd machinery — "everything is a file descriptor" in Unix. Each process has its own table — fd 5 in process A is unrelated to fd 5 in process B.
- **Signal handlers** — each process has an array of functions indexed by signal number. When you Ctrl-C, the kernel delivers SIGINT (signal 2); the registered handler runs. Python's default SIGINT handler raises `KeyboardInterrupt`. Set custom handlers via `signal.signal(...)`. Handlers are per-process — a parent's handlers don't carry to its child after exec.
- **Other per-process state**: current working directory, user/group ID (permissions), environment variables (yes, `os.environ` is per-process), resource limits (`ulimit`), scheduling priority.

*Pipes.* A pipe is an in-memory, kernel-managed FIFO buffer. Write to one end (an fd), read from the other. Unidirectional. Created with the `pipe()` syscall, which returns two fds (read end, write end). Pipes live in kernel memory, not on disk — fast.

When you run `ls | grep foo` in a shell: shell creates a pipe, dups `ls`'s stdout (fd 1) to the write end, dups `grep`'s stdin (fd 0) to the read end, kernel handles buffering.

In `subprocess.Popen(stdout=PIPE)`, Python creates a pipe, gives the child the write end as its fd 1, gives you the read end as `proc.stdout`. That's how a parent captures a child's output.

TraceML doesn't use pipes for telemetry (it uses TCP), but child processes do inherit the parent's stdio by default — which is how training's printed output shows up in your terminal.

*Network sockets.* A socket is a kernel-managed bidirectional communication endpoint. Think of it as a two-way pipe that can span machines (TCP/IP) or stay local (Unix domain sockets, or TCP on localhost). Create with `socket()` → fd. `bind()` assigns an address; `listen()` makes it a server; `accept()` blocks until a client connects; `connect()` is the client side.

On localhost (127.0.0.1), a TCP socket still routes through the kernel but never touches the network card. Latency is microseconds. TraceML uses localhost TCP between ranks and aggregator — same machine, but TCP gives clean bidirectional streams without shared-memory concurrency headaches.

*"Length-prefixed msgpack" explained.* TCP is a **byte stream**, not a message protocol. If the sender does `send(b'hello')` then `send(b'world')`, the receiver's `recv()` might return `hellowor` on one call and `ld` on the next — TCP doesn't preserve message boundaries. Solution: prefix each message with its length (4 bytes), then payload. Receiver reads 4 bytes, learns N, reads N bytes, decodes as one message. This is "framing." TraceML uses 4-byte length + msgpack payload.

**Concepts introduced:** kernel, syscall, user vs kernel mode, virtual memory, page table, MMU, file descriptor, address space, `task_struct`, pipe (kernel FIFO), socket, TCP localhost, byte stream vs message protocol, length-prefixed framing.

---

### Q7: Spawning — fork, exec, and multiprocessing start methods

**Date:** 2026-04-24
**Source:** expansion of "How spawning actually works on Linux" from Q1, plus "what is spawning"

**Short answer:** "Spawning" = launching a new process. On Linux, the low-level mechanism is a two-step dance: **fork** (duplicate myself, via `clone`) + **exec** (replace my code with a different program, via `execve`). Fork creates a near-identical child using copy-on-write memory; exec turns that child into a different executable. `subprocess.Popen` chains them for you. Python's `multiprocessing` has three **start methods** — `fork`, `spawn`, `forkserver` — picking when to use each is critical when mixing with CUDA or threads.

**Long answer:**

*The fork syscall.* `fork()` is the kernel call that creates a new process. It's famously weird: it returns *twice* — once in the parent (returning the child's PID), once in the child (returning 0). That's how the branches know which one they are:

```python
pid = os.fork()
if pid == 0:
    # child branch
    ...
else:
    # parent branch, pid = child's PID
    ...
```

After fork, the child is a near-exact clone of the parent: same memory contents, same open file descriptors, same cwd, same env vars, same Python interpreter with all its loaded modules.

Memory is **copy-on-write (COW)**. The kernel doesn't actually duplicate gigabytes of pages. Both processes start sharing the same physical pages, marked read-only in their page tables. When either side writes, the MMU faults, the kernel copies that one page, updates the writer's page table to point at the copy, marks both writable. Lazy duplication → fork is fast even for huge processes.

Internally in modern Linux, fork is implemented via the more general `clone()` syscall. `clone()` takes flags that control what gets shared (memory, fds, signals) and what's copied. Same flags with different settings give you threads vs processes — they're really the same primitive.

*The exec family.* `execve(path, argv, env)` tells the kernel: take my current process (same PID) and replace its entire code + memory with a new program. It:

1. Validates the target executable (ELF header, permissions).
2. Wipes the calling process's memory mappings — old code, stack, heap, globals — gone.
3. Loads the new program's code and data into the now-empty address space.
4. Jumps to the new program's entry point (`_start`, which calls `main`).

PID stays the same. Open file descriptors survive by default (unless marked close-on-exec, `FD_CLOEXEC`). Environment variables come from the `env` argument. On success, exec never returns — the caller has been replaced. It only returns on failure (bad path, permission denied).

*Why two steps?* Flexibility. Between fork and exec, the child can: redirect pipes, change user ID (privilege drop), chroot, set cgroup, close inherited fds. This is how shells implement `|`, `<`, `>`, and how containers launch isolated processes.

```python
pid = os.fork()
if pid == 0:
    os.dup2(read_end, 0)   # child: replace stdin with a pipe
    os.execvpe("/bin/ls", ["ls", "-la"], custom_env)
    os._exit(1)            # only reached if exec fails
else:
    os.waitpid(pid, 0)
```

*Python's `subprocess.Popen` under the hood.* Does a fork (via `clone`), sets up any pipes you requested for stdin/stdout/stderr, executes the target with `execve`, returns a `Popen` handle in the parent (`.pid`, `.wait()`, `.communicate()`). If the target is a Python script (like TraceML's aggregator), you're execing `/usr/bin/python3` with your script as argv[1] — fresh Python interpreter, no inherited Python state.

*Python's `multiprocessing` start methods.* When you do `multiprocessing.Process(target=f)`, Python has three strategies:

1. **"fork"** (historical default on Linux): just fork, no exec. Child inherits all parent state — imports, globals, loaded libraries, open fds, logging handlers. Fast. But DANGEROUS:
  - If parent had multiple threads at fork time, only the forking thread survives in the child — but it inherits all the mutexes in whatever state they were in. Can deadlock. See "fork safety" in CPython docs.
  - **CUDA contexts are not fork-safe.** Forking after initializing CUDA in the parent → broken CUDA in the child. Very common ML footgun. Fix: fork before CUDA init, or use "spawn".
2. **"spawn"** (default on macOS since Python 3.8, default on Windows always, becoming default on Linux in 3.14 for `multiprocessing`): fork + exec of a clean Python interpreter + re-import the target module + deserialize the arguments. Slow startup (hundreds of ms), but clean.
3. **"forkserver"**: fork once early into a minimal "server" process (before loading CUDA, before threading). Each subsequent worker forks from that minimal server. Fast-ish and CUDA-safe. Often the right choice for ML.

*How torchrun spawns training workers.* `torchrun` is itself a Python CLI. It fork+execs N Python subprocesses (one per `--nproc_per_node`). Before exec, it sets env vars `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` in each child. Each child then imports PyTorch fresh, eventually calls `torch.distributed.init_process_group(...)`, reads those env vars, joins the collective. Every rank is a brand-new interpreter — no shared memory at startup — which is why env vars are the configuration mechanism.

*Where this shows up in TraceML.* In [traceml/src/traceml/cli.py](traceml/src/traceml/cli.py), two `subprocess.Popen` calls:

- One spawns `python -m traceml.aggregator_main ...` — fresh Python, fresh aggregator state.
- One spawns `torchrun ... train.py` — which itself forks+execs N training ranks.
- Both use `start_new_session=True`: creates a new process group so a single `os.killpg(pgid, SIGTERM)` on Ctrl-C tears down the whole tree atomically.

**Concepts introduced:** spawning, fork, exec (execve), clone, COW, ELF, FD_CLOEXEC, fork-safety, CUDA fork-safety, multiprocessing start methods (fork/spawn/forkserver), torchrun internals, process group, session, rendezvous.

---

## Networking

### Q10: What is TCP concretely, and what's a port?

**Date:** 2026-04-24
**Source:** suggested follow-up after Q5 — fills out the network picture

**Short answer:** **TCP** (Transmission Control Protocol) is a transport-layer protocol that gives you a reliable, ordered, bidirectional stream of bytes between two endpoints. A **port** is a 16-bit integer (0–65535) the kernel uses to multiplex many simultaneous connections on the same machine — IP address tells the kernel *which machine*, port tells it *which socket on that machine*. The pair (protocol, src IP, src port, dst IP, dst port) — the 5-tuple — uniquely identifies a connection.

**Long answer:**

*The networking stack layers (top-down).* Every byte your process sends travels through layers of encapsulation:

- **Application layer** — your protocol (HTTP, msgpack-over-TCP, gRPC, TraceML's framed messages). What you write.
- **Transport layer** — TCP or UDP. Adds source/dest port, sequence numbers, acknowledgments, checksums.
- **Network layer** — IP (v4 or v6). Adds source/dest IP address, routes packets across networks.
- **Link layer** — Ethernet, Wi-Fi. Adds MAC addresses, handles physical transmission.

When you do `socket.send(b'hello')`, the kernel wraps `hello` in a TCP segment, then an IP packet, then an Ethernet frame, then sends it on the wire. The receiver's kernel unwraps in reverse and delivers `hello` to the receiving socket's read buffer. All of this is per-syscall — Python orchestrates, the kernel does the encapsulation.

*What TCP guarantees.*

- **Reliable**: dropped packets are retransmitted. Bit-flipped packets are detected via checksums and discarded/retried.
- **Ordered**: bytes arrive at the application in the order they were sent, even if individual packets took different network paths and arrived out of order.
- **Stream-oriented**: there's no notion of message boundaries — just a stream of bytes flowing in each direction.
- **Bidirectional (full-duplex)**: both sides can send simultaneously.
- **Connection-oriented**: you do a handshake to establish a connection (the famous three-way handshake: SYN → SYN-ACK → ACK), maintain it via sequence numbers and ACKs, tear it down with FIN/RST.

What TCP does *not* guarantee: message framing (you have to do this yourself — see Q5 on length-prefixing), or low latency (TCP can pause sending for congestion control, leading to "head-of-line blocking" when a single packet is lost — all subsequent bytes wait for the retransmit).

*Contrast with UDP.* UDP is the other big transport protocol: connectionless, no retransmission, no ordering, but message-oriented (each `sendto` produces one datagram preserved as a whole; either you receive it all or not at all). Use TCP when you need reliability (most app protocols). Use UDP when retransmits are pointless (real-time gaming, voice, video, DNS lookups). NCCL across nodes typically uses RDMA or TCP, not UDP.

*Ports.*

- A port is a 16-bit integer (0–65535) the kernel uses to multiplex sockets on one machine.
- An IP packet has source IP, dest IP. The TCP segment inside it adds source port, dest port. The **5-tuple** (protocol, src IP, src port, dst IP, dst port) uniquely identifies a connection on a machine. The kernel uses this tuple to demultiplex incoming packets to the right socket.
- Ports 0–1023 are "well-known" (HTTP=80, HTTPS=443, SSH=22). Binding to these requires root on Unix.
- Ports 1024–49151 are "registered" (assigned by IANA but you can use them in dev).
- Ports 49152–65535 are "ephemeral" — the kernel automatically picks one for the client side when you `connect()` without explicitly binding.
- A server `bind()`s to a specific port (e.g., 8000) so clients can find it. Many clients can connect to the same server port — each connection gets a unique 5-tuple because the *client* port is ephemeral and different. The kernel demultiplexes incoming packets to the right socket using the full 5-tuple.

*Localhost (127.0.0.1).* The kernel has a "loopback interface" (`lo`) — a virtual network device pointing to the same machine. Packets sent to 127.0.0.1 (or `::1` for IPv6) never go to the actual network card; they're handled internally by the kernel. Same TCP semantics still apply (reliable, ordered, framing-free), but latency is microseconds, not milliseconds. No real checksum errors, no real retransmits — but the kernel still runs the full TCP state machine. This is why TraceML can use TCP for ranks-to-aggregator without paying real network costs.

*TraceML's specific use.*

- The aggregator does `socket()`, `bind(("127.0.0.1", port))`, `listen()`, `accept()` in a loop. Each rank does `socket()`, `connect(("127.0.0.1", port))`, `send()`.
- Why TCP and not a Unix domain socket or shared memory? Because TCP keeps the door open to multi-node — same code with a different IP would let ranks on different machines connect to a central aggregator. Unix domain sockets would be slightly faster (skip even the loopback stack) but lock you to one machine. Shared memory would be even faster but introduces concurrency complexity that fights the fail-open design.
- Port is dynamically chosen at startup to avoid conflicts with other processes. The CLI writes the chosen port into the manifest (or env var) so each rank knows where to connect. Probing for port readiness is part of the CLI's startup dance.

**Concepts introduced:** TCP, UDP, IP, port (16-bit), 5-tuple connection identity, well-known/registered/ephemeral port ranges, three-way handshake, full-duplex, byte stream vs datagram, loopback interface (`lo`), 127.0.0.1 / ::1, OSI/Internet layer model, encapsulation, head-of-line blocking, congestion control, MTU, kernel network stack, Unix domain sockets vs TCP-on-localhost.

---

### Q14: What is RDMA / Infiniband and why does it matter for multi-node training?

**Date:** 2026-04-24
**Source:** suggested follow-up after Q12 — natural deepening of the multi-node side of NCCL all-reduce

**Short answer:** **RDMA** (Remote Direct Memory Access) is a network protocol where one machine writes/reads directly into another machine's memory without involving the remote CPU or kernel. **Infiniband** is a network fabric (cables, switches, NICs) designed around RDMA from the ground up. Together they enable multi-100 Gb/s inter-node communication at single-digit microsecond latency — necessary to keep multi-node training collectives from being the bottleneck. Plain TCP can't sustain those rates because of syscall, copy, and protocol-processing overhead in software.

**Long answer:**

*Why TCP is too slow at scale.* A modern multi-node training cluster has 200–800 Gb/s NICs per node. To shift 350 MB of gradients within a 10 ms training-step budget you need ~280 Gb/s effective. TCP can't do that easily because:

- **Syscall overhead**: every `send`/`recv` costs microseconds. Many small writes dominate.
- **Memory copies**: data is copied from user-space buffer → kernel socket buffer → NIC DMA buffer. Each copy is hundreds of ns/MB at memory bandwidth.
- **Interrupt handling**: NIC fires interrupts, kernel runs softirq handlers, schedules user thread.
- **Protocol processing**: TCP state machine, ACKs, congestion control, sequence numbers — all in software, all on the CPU.

At 100 Gb/s+, the CPU spends most of its time on networking. You burn cycles on the very work you wanted to free for actual computation.

*What RDMA does.* RDMA pushes the protocol into the NIC hardware. Key principles:

- **Kernel bypass**: user applications submit work directly to the NIC via mapped memory (queue pairs). No syscalls in the data path.
- **Zero-copy**: the NIC DMAs from registered user buffers directly. No kernel buffer copy.
- **One-sided operations**: an "RDMA WRITE" sends data to a remote address — the remote CPU is not involved at all. The remote NIC writes to RAM and (optionally) notifies via a separate completion queue.
- **Hardware offload**: connection state, retransmits, ordering — all in NIC silicon.

The programming interface is **verbs** (libibverbs on Linux). You allocate "queue pairs" (a send queue + receive queue), register memory regions, and post work requests. Frameworks like NCCL use verbs under the hood — you don't write verbs code yourself in normal training.

*Infiniband.* Infiniband is a separate networking fabric — different cables (QSFP), different switches, different NICs from Ethernet. Designed from the ground up for RDMA, low latency, and **lossless** transmission (credit-based flow control instead of TCP-style retransmission).

- Latency: ~1 μs end-to-end for small messages.
- Bandwidth: up to 800 Gb/s with NDR Infiniband (2024 generation).
- Deployments: HPC clusters, large GPU farms, NVIDIA's DGX SuperPOD reference architecture.

*RoCE (RDMA over Converged Ethernet).* RoCE runs RDMA verbs over Ethernet hardware. RoCEv2 (the common version) encapsulates the RDMA payload in UDP packets, so it's routable across IP networks. Why this matters: Ethernet is cheaper, more universal, and you don't have to build a separate fabric. Performance is competitive with Infiniband but requires careful network configuration (PFC, ECN) for lossless operation.

*GPUDirect RDMA.* Standard RDMA path on a multi-GPU node: GPU → host RAM → NIC → wire → remote NIC → remote host RAM → remote GPU. That's two GPU↔CPU copies plus two NIC↔RAM copies.

GPUDirect RDMA: GPU memory is exposed via PCIe peer-to-peer to the NIC. The NIC DMAs *directly* from GPU memory. So: GPU → NIC → wire → remote NIC → remote GPU. Saves four memory copies, roughly halves latency. NCCL uses this on supported hardware (most modern HPC GPUs + NICs).

*How NCCL uses it.*

- Intra-node: NVLink for GPU↔GPU peer transfers, no NIC involved.
- Inter-node: NCCL transparently uses verbs over Infiniband / RoCE / TCP fallback. Picks the best available transport at `init_process_group` time.
- NCCL's all-reduce algorithm (Q12) runs the same way regardless of transport — just with different bandwidth and latency profiles.

*Why this matters for distributed training.*

- A 70B-parameter model has ~280 GB of gradients to all-reduce per step in pure data parallelism. *Physically impossible* to move that within a step budget without high-bandwidth interconnect.
- For mixtures of techniques (FSDP, ZeRO, tensor-parallel, pipeline-parallel), each one trades compute time for communication time differently. The right choice depends critically on your network: with 800 Gb/s NDR Infiniband + GPUDirect, comm cost is low and you can fully shard parameters; with 100 Gb/s Ethernet without RDMA, you have to keep more local and accept memory pressure.
- This is why HPC-style clusters (Infiniband, NVLink, GPUDirect) crush commodity Ethernet clusters at the same nominal GPU count for training large models.

*Why TraceML doesn't directly use RDMA.* TraceML's TCP traffic is rank → aggregator on localhost — single-machine, low-volume. Plain TCP is fine and fail-safe. But TraceML's *diagnostics* could surface "your collective time is limited by your interconnect." If rank step time is dominated by NCCL all-reduce on a low-bandwidth fabric, the user benefits from knowing that the comm fabric is the bottleneck, not GPU compute or dataloader. This is on the roadmap (sales-hook territory) rather than implemented.

*Future TraceML feature ideas.* Surface the NCCL transport choice (TCP vs IB vs RoCE), surface measured per-step bandwidth during runs, compare against fabric peak, flag when comm-bound runs would benefit from a fabric upgrade.

**Concepts introduced:** RDMA, kernel bypass, zero-copy, one-sided operations, queue pairs, verbs API (libibverbs), Infiniband fabric, lossless transport via credit-based flow control, RoCE v1/v2, PFC/ECN configuration, GPUDirect RDMA, NIC PCIe peer-to-peer, why TCP doesn't scale to 100+ Gb/s, NCCL transport selection, comm-bound vs compute-bound training, NDR Infiniband generation.

---

## Python Internals

### Q6: Python internals — bytecode, GIL, CPU-bound vs I/O-bound

**Date:** 2026-04-24
**Source:** clarifications on Q1 ("Python bytecode at a time", "CPU-bound Python parallelism", "N training workers each running a forward pass concurrently")

**Short answer:** Python source compiles to **bytecode** — stack-based instructions for the CPython virtual machine. The **GIL** (Global Interpreter Lock) is a mutex that ensures only one thread executes bytecode at once in a given process. **CPU-bound** Python code (tight pure-Python loops) can't be parallelized with threads because of the GIL. **I/O-bound** code can, because blocking syscalls release the GIL during the wait. C extensions like NumPy and PyTorch release the GIL for heavy math — which is why multithreaded dataloaders work.

**Long answer:**

*Python bytecode.* When you run `python foo.py`, the compiler turns source into a list of low-level instructions called **bytecode** (e.g., `LOAD_CONST`, `BINARY_ADD`, `STORE_FAST`, `CALL_FUNCTION`). You can inspect it: `import dis; dis.dis(my_func)`. Bytecode runs on the **CPython virtual machine** — a giant `while True: switch(opcode)` loop in C (in `ceval.c`). It's "bytecode" because each instruction is a short byte sequence. This differs from "compiling to machine code" (what C, Rust, Go do). Bytecode is portable across CPU architectures but interpreted. Context: JIT languages (V8 for JavaScript, PyPy for Python) go further and compile hot bytecode to native code at runtime. Regular CPython doesn't (CPython 3.14 has experimental JIT).

*The GIL.* CPython uses **reference counting** for garbage collection — every object carries a count of who's pointing at it. Every touch bumps the count up or down. If two threads incremented the same refcount concurrently without synchronization, you'd get lost updates → memory corruption. The pragmatic 1992 solution: wrap every bytecode step in a single global mutex. One thread at a time holds the GIL and executes bytecode; others wait. Threads rotate (~every 100 bytecodes or on voluntary yield).

Consequences:

- **CPU-bound Python code doesn't scale with threads.** Two threads running a tight pure-Python loop on a 4-core machine take ~2× as long as one thread (not 0.5× as you'd hope) — they serialize on the GIL and pay context-switch overhead on top.
- **I/O-bound Python code *does* scale with threads.** Blocking syscalls (reading from a socket, waiting on disk) release the GIL while blocked. Another thread can grab the GIL and run Python during that wait. This is why web servers with many threads can serve many concurrent requests.
- **C extensions can release the GIL explicitly.** NumPy, PyTorch, TensorFlow all release the GIL while doing their heavy math in C/C++/CUDA. So `y = x @ W` on a big tensor drops the GIL for the matmul's duration — another Python thread can run during that window. This is why PyTorch's DataLoader with `num_workers=0` still benefits from threading for some decode operations (though `num_workers>0` uses subprocesses for full parallelism).
- **Bypassing the GIL means using processes.** Each process has its own Python interpreter, its own GIL. N processes = N truly-parallel Python interpreters. `multiprocessing`, `concurrent.futures.ProcessPoolExecutor`, `subprocess`, torchrun — all use this strategy.
- **The future.** Python 3.13 has an experimental "free-threaded" (no-GIL) build (PEP 703). Still immature; most libraries haven't been audited for thread-safety without it. Production ML still assumes one GIL per process.

*CPU-bound vs I/O-bound.*

- **CPU-bound**: bottleneck is computation. CPU pinned at 100%. More cores help only if you can parallelize. Examples: a tight Python loop counting primes, a pure-Python matmul.
- **I/O-bound**: bottleneck is waiting for something external. CPU idle during the wait. Examples: fetching 1000 URLs, reading from disk, waiting on a socket.
- For ML: training is **GPU-bound** (a subtype of compute-bound — GPU saturated, CPU mostly waiting on GPU). Dataloading can be I/O-bound (disk) or CPU-bound (JPEG decode, augmentation). That's why PyTorch's DataLoader uses `num_workers > 0` — subprocesses to parallelize decode past the GIL.

*Why this demands one-process-per-GPU for distributed training.* With N GPUs, you want N forward passes happening in parallel. Each forward pass runs Python (layer orchestration), calls into PyTorch C++ (which releases GIL, launches CUDA kernels, awaits completion). If all N forward passes lived in one process as threads:

- The Python orchestration of each forward would serialize on the GIL.
- With N=8 GPUs, you'd see ~8× the Python overhead on the critical path — significant at small batch sizes.
- You'd also conflate CUDA contexts and have coordination headaches. With one process per GPU, each process owns one CUDA context, its own GIL, its own Python-side dispatcher — fully parallel, no cross-contamination.

**Concepts introduced:** bytecode, CPython VM (`ceval.c`), reference counting, GIL, mutex, CPU-bound vs I/O-bound, GIL-releasing C extensions, multiprocessing, PEP 703 / free-threaded Python, GPU-bound, why one-process-per-GPU.

---

### Q9: What are hooks, and what does "injecting hooks in-process" mean?

**Date:** 2026-04-24
**Source:** clarification on "TraceML injects its hooks in-process alongside your code"

**Short answer:** A **hook** is a user-supplied callback registered with a library so the library invokes it at specific lifecycle events (e.g., "after forward pass," "before optimizer step"). **Injecting hooks** = TraceML attaches these callbacks to *your* PyTorch model/dataloader at runtime, without modifying your source code. Combined with **monkey-patching** (replacing library functions at runtime), this is how TraceML observes your training loop without you ever importing or calling a TraceML function inside it.

**Long answer:**

*The general hook concept (not PyTorch-specific).* A hook = a registered callback. The library publishes lifecycle events; your callback subscribes. Examples outside ML: React's `useEffect`, Git's `pre-commit` hook, bash's `trap`, JVM bytecode transformers, Chrome's `onBeforeRequest`. In RL, similar to an env wrapper that intercepts `step()` — your wrapper doesn't own the env but gets called around each transition.

*PyTorch's built-in hook system.* PyTorch exposes hooks on `nn.Module` objects:

- `module.register_forward_pre_hook(fn)` — called right before the module's forward.
- `module.register_forward_hook(fn)` — called right after.
- `module.register_full_backward_hook(fn)` — called around backward.
- `tensor.register_hook(fn)` — called when a gradient is computed for that tensor.
- Each returns a handle; `handle.remove()` detaches the hook.

Internally, `nn.Module.__call__` (the thing that runs when you do `out = model(x)`) checks for registered hooks and fires them around the actual forward. These hooks let you observe or modify activations/gradients without changing the model's class. Standard uses: visualizing activations, gradient clipping by layer, pruning, profiling.

*How TraceML uses hooks.* In [traceml/src/traceml/decorators.py](traceml/src/traceml/decorators.py), `trace_model_instance(model)` walks the module tree and attaches forward and backward hooks on each layer. The hooks:

- Record the current time (via CUDA events for GPU work, or `time.perf_counter` for CPU).
- Record peak memory allocation delta.
- Append a row to the relevant sampler's `Database`.

So when your training loop runs `loss = model(batch)`, PyTorch fires TraceML's hooks on every layer as it executes — **in the same thread, same Python process**. That's what "in-process" means: no IPC, no thread boundary. The hook callback is just extra Python code running immediately before/after each layer.

*What "injection" means here.* Two distinct mechanisms together:

- **Hook injection** — registering `register_forward_hook` callbacks on your model *instance* at runtime. You call `trace_model_instance(model)` once; from then on your specific model instance has TraceML hooks. You didn't modify the model's class definition; the hooks are attached to the instance in memory.
- **Monkey-patching (a.k.a. patching)** — replacing a function/method at runtime. For example, TraceML's dataloader patching in [traceml/src/traceml/utils/](traceml/src/traceml/utils/) wraps `DataLoader.__iter__`. Your code still calls `for batch in loader:`, but `__iter__` now points to TraceML's wrapper, which calls the original iterator and adds timing around it.

Why this is powerful for observability: the user doesn't need to refactor train.py. They just run `traceml run train.py` (or `import traceml`), and TraceML attaches itself — hooks on the model, patches on PyTorch's dataloader/optimizer.

*Concrete zero-code flow.*

1. User runs `traceml run train.py`.
2. CLI spawns aggregator (out-of-process) and torchrun (in-process, N training ranks).
3. Each rank, *before* running train.py, executes TraceML's runtime setup, which:
  - Monkey-patches `DataLoader.__iter__`, certain `nn.Module` entry points, etc.
  - Starts sampler threads.
  - Opens a TCP connection to the aggregator.
4. Train.py runs. When the user calls `model(x)`, the patched call fires TraceML hooks, which record timing and memory, which go into the Database, which `DBIncrementalSender` ships over TCP. All transparent to the user's training code.

*Why this matters in the architecture.* The diagram's line *"Zero-code: one `import traceml`, or `trace_step(model)`"* is made possible by hook injection + monkey-patching. Without them, you'd need to sprinkle TraceML calls across your training loop. Instead, TraceML becomes a passive observer that doesn't need the user's code to know about it.

*Failure modes to be aware of (useful for mental model).*

- Hooks only fire for modules you hooked. If a user does custom operations in `torch.`* (not inside an `nn.Module`), those are invisible to module-level hooks. Monkey-patching the right functions is how you'd catch them.
- Monkey-patches can conflict: if another library also patches the same function, order matters. This is part of the reliability work that's called out in TraceML's constraints ("PyTorch coupling... test coverage is critical").

**Concepts introduced:** hook (callback pattern), observer pattern, PyTorch `register_*_hook`, monkey-patching, handle (for hook removal), in-process instrumentation, zero-code instrumentation, module instance vs class, patching conflicts.

---

## CUDA & GPU

### Q11: What is a CUDA context, and why is it fork-unsafe?

**Date:** 2026-04-24
**Source:** suggested follow-up after Q7 — anchors all CUDA-adjacent discussions

**Short answer:** A **CUDA context** is a per-process state object the CUDA driver creates to track GPU resources owned by that process: memory allocations, streams, loaded kernel modules, current device, locks. It's roughly the GPU-side equivalent of a process's address space. It is **fork-unsafe** because the CUDA driver maintains in-process bookkeeping (locks held by helper threads, file descriptors to `/dev/nvidia`*, host-side queue buffers) that doesn't survive fork's "duplicate the process and drop all but the calling thread" semantics — both parent and child end up referring to the same kernel-side driver state, with the child's locks in a partially-held state, causing deadlocks or silent corruption.

**Long answer:**

*What's in a CUDA context.* When you do `import torch; torch.zeros(1).cuda()` (or any CUDA op), the CUDA runtime/driver lazily creates a context for your process. The context owns:

- **GPU memory allocations** — every `cudaMalloc` is tracked back to its owning context. The driver keeps a per-context memory map.
- **Streams** — ordered queues of GPU work. `torch.cuda.current_stream()` returns one. Operations submitted to the same stream run in order; different streams can overlap.
- **Loaded modules / kernels** — when you call `y = torch.relu(x)`, PyTorch ensures the relu CUDA kernel is loaded into the context. Modules are PTX or cubin (compiled CUDA bytecode/binary).
- **Current device** — `torch.cuda.set_device(local_rank)` updates the context's notion of "what device am I targeting?"
- **Pinned host memory regions, host-side command buffers, queue state** — used for fast CPU↔GPU transfers and kernel submission.
- **Synchronization primitives** — locks, events, fences for coordinating CPU/GPU work.

The context lives partly in user-space (driver libraries you've loaded into your process — `libcuda.so`) and partly in kernel-space (held by the GPU's kernel driver, typically `nvidia.ko` on Linux). The user-space side talks to the kernel side through file descriptors opened by libcuda — `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`.

*Why context exists at all.* Multiple processes can share one GPU. The driver needs per-process bookkeeping so it can: (a) keep memory allocations from one process invisible to another, (b) round-robin scheduling of kernels from different processes, (c) clean up when a process dies (reclaim memory, release locks). Without contexts, two processes touching the same GPU would step on each other.

*Why fork breaks this.* Recall fork (Q7): child gets a duplicate address space (copy-on-write), inherits open file descriptors, but only the calling thread survives — every other thread in the parent vanishes in the child.

Now consider the parent had an active CUDA context. After fork:

1. The child inherits all the user-space libcuda state — pointer values, allocator structures, lock state. Looks like the child has its own context, but the underlying state is shared/inconsistent.
2. The child also inherits the open fds to `/dev/nvidia`*. Those fds in the kernel point to the *parent's* context. Any CUDA call from the child either interferes with the parent's GPU work or gets a "context corrupt" error.
3. libcuda has internal helper threads holding locks. After fork those threads don't exist in the child, but the locks they held are still marked held. The child eventually hits one of those locks → deadlock forever.
4. Even ignoring locks, the GPU driver's command queues, in-flight transfers, and event objects are now ambiguously owned. Submitting from the child can corrupt the parent's pending work.

In practice you see errors like:

- `CUDA error: initialization error`
- `CUDA error: invalid device context`
- Hangs in `cuInit` or memory allocation
- Silent data corruption (rare but worst case)

*Why "spawn" or "forkserver" fixes it.*

- **spawn**: fork + immediate exec of a fresh Python interpreter. The exec wipes the parent's address space entirely (Q7). The child then imports CUDA fresh from scratch, creating its own clean context. Slow startup, but correct.
- **forkserver**: fork once early, before any CUDA imports. Keep that minimal "server" process around. Subsequent worker forks come from the minimal server (which has no CUDA state) → safe, faster than spawn.

*PyTorch-specific landmines.*

- Calling `torch.cuda.is_available()` initializes a context in some configurations. People often do this at module top level.
- `init_process_group` for DDP initializes NCCL, which initializes CUDA. Don't fork after.
- `DataLoader(num_workers > 0)` uses multiprocessing. Default start method historically was "fork" → footgun with CUDA. PyTorch defaults are evolving, and the recommended pattern is to set up CUDA only after worker processes have started, or to use `multiprocessing.set_start_method("spawn")`.
- Classic anti-pattern: training script does `model.cuda()` in the parent, then `multiprocessing.Pool(workers=4)` for parallel data prep — boom, broken CUDA in the workers.

*Why this matters for TraceML.*

- TraceML's aggregator process is spawned by the CLI *before* any CUDA initialization happens — it just talks TCP. Safe.
- Training ranks are launched via torchrun, which fork+execs (effectively spawn-style) — fresh interpreters per rank, no CUDA fork issue.
- TraceML's monkey-patching runs *inside* each training process (after the fork+exec). It doesn't fork. So TraceML never triggers a fork-unsafe path.
- If TraceML ever wanted background workers inside a training process, it'd have to use `set_start_method("spawn")` to be CUDA-safe.

*Mental model.* Think of a CUDA context as analogous to a database connection pool: stateful, has locks and worker threads, lives partly inside your process and partly on the server (the GPU). Forking a process while holding an active connection pool would similarly break — handles point at server-side state that now has two owners. Fix is the same: open a fresh pool in each child.

**Concepts introduced:** CUDA context, libcuda.so, `/dev/nvidia`* device files, kernel-mode GPU driver (nvidia.ko), GPU memory allocator, CUDA streams, kernel modules (PTX/cubin), multi-process GPU sharing, fork-unsafety mechanics (locks held by vanished threads), spawn/forkserver remediation, PyTorch DataLoader fork pitfall, MPS (multi-process service) as a related but separate topic, connection-pool analogy.

---

### Q15: What is a CUDA stream and how does it differ from a CPU thread?

**Date:** 2026-04-24
**Source:** suggested follow-up after Q11 — pairs with the CUDA context discussion

**Short answer:** A **CUDA stream** is an ordered FIFO queue of GPU work. Operations submitted to the same stream run in submission order; operations on different streams can run concurrently on the GPU. It is **not** an execution context — it has no stack, no program counter, no scheduler thread. A **CPU thread**, by contrast, is a real execution context with its own stack, scheduled by the OS, subject to the GIL in Python. Streams are work *channels*; threads are workers. You can think of a stream as a pipe to a remote job queue (the GPU) and the GPU's hardware scheduler as the worker pool.

**Long answer:**

*What a CUDA stream actually is.* A CUDA stream is a handle to an in-driver queue of asynchronous operations. When you do `y = torch.relu(x)`:

1. PyTorch's dispatcher calls into the CUDA kernel implementation.
2. The CUDA runtime *queues* a kernel-launch operation onto the current stream.
3. Returns immediately to Python — the kernel hasn't run yet.
4. The GPU's hardware scheduler eventually picks up that work and runs the kernel on its compute units (SMs — Streaming Multiprocessors).
5. The result becomes valid in `y` only when the kernel completes — but Python continues executing immediately.

This is asynchronous from the host CPU's perspective. The host can queue more work without waiting for completion. **The stream is just the queue handle**; the actual execution happens on the GPU.

*Operations on the same stream are ordered.* By definition: two ops queued on stream S run sequentially. The GPU starts the second only after the first's writes are visible. This is what gives you correctness — `y = relu(x); z = y * 2` works because both ops go on the same stream by default.

*Operations on different streams can run concurrently.* The GPU has many SMs (an A100 has 108, an H100 has 132). If stream A has a small kernel using 4 SMs and stream B has a kernel using 8 SMs, both can run simultaneously on the same GPU. This is "concurrent kernel execution" and is the primary reason for using multiple streams: hide latency by overlapping independent work.

Common pattern: one stream for compute, another for H2D (host-to-device) memcpy. While the GPU is computing on batch N, the next batch is being copied from CPU RAM in parallel. This is what PyTorch's `non_blocking=True` argument on `tensor.cuda()` enables when paired with pinned memory.

*Default stream gotchas.*

- Historically (CUDA ≤ 6), the "default stream" (stream 0) was implicitly synchronizing — work on it blocked all other streams. Bad for concurrency.
- Now there's "per-thread default stream" (compile-time flag) — each CPU thread gets its own default stream, so different threads can queue work without false serialization.
- `torch.cuda.set_stream(s)` lets you explicitly use stream `s` for subsequent ops on the current thread.
- `with torch.cuda.stream(s):` is a context manager wrapper around the same.

*Why streams ≠ CPU threads.*


|                     | CPU thread                     | CUDA stream                    |
| ------------------- | ------------------------------ | ------------------------------ |
| What it is          | Execution context              | Work-queue handle              |
| Has program counter | Yes                            | No                             |
| Has stack           | Yes                            | No                             |
| Scheduled by        | OS kernel                      | GPU hardware scheduler         |
| Subject to GIL      | Yes (in Python)                | No (GIL is host-side only)     |
| Synchronization     | locks, condition vars, atomics | stream APIs, CUDA events       |
| Concurrent count    | bounded by cores               | bounded by SM count + memory   |
| Failure mode        | crash, deadlock                | wrong ordering, race in memory |


You can have one CPU thread queueing work on multiple streams. You can also have multiple CPU threads queueing on the same stream. The mapping is many-to-many.

*Synchronization between streams: CUDA events.* A `cudaEvent_t` is a marker you can record on a stream. You can:

- `event.record(stream)` — insert the event into stream's queue. It "fires" when all work queued before it has completed.
- `stream.wait_event(event)` — make stream's future work wait for this event to fire.
- `event.synchronize()` — block the host CPU until the event has fired.
- `event.elapsed_time(other_event)` — measure GPU time between two events on the same stream (in milliseconds).

*How TraceML uses streams and events.* TraceML's layer-time sampler records two CUDA events per layer per step: one before forward, one after. After the step completes (or asynchronously), TraceML calls `event.elapsed_time(start, end)` to get the actual GPU duration. **This is the only way to measure GPU time accurately** — `time.perf_counter()` measures host-side dispatch time, which is just "when did Python queue this work?", not "when did the GPU finish it?"

The CUDA event pool in [traceml/src/traceml/utils/](traceml/src/traceml/utils/) reuses event objects across steps to avoid allocation cost. (Creating a CUDA event is cheap but not free, ~microseconds.)

*Why this matters for understanding bottleneck diagnostics.*

- If TraceML reports a layer's "host time" >> "GPU time", the bottleneck is Python dispatch, not GPU work — you're CPU-bound on the framework.
- If "GPU time" >> "host time", the host queues work fast but the GPU is the bottleneck — you're compute-bound; accelerate the kernel or use a bigger GPU.
- If both are small but step time is big, you're stuck waiting on something else (dataloader, all-reduce barrier, kernel queue stall).

This decomposition only makes sense once you understand that streams and threads are different abstractions. The CPU thread pumps Python; the stream queue holds the actual work; the GPU schedules and executes asynchronously; events are the bridge that lets you measure when GPU work actually happened.

*Mental model.*

- **CPU thread** = a worker. Has its own scratch space (stack), gets shoved around by the OS scheduler, occasionally talks to other threads via synchronization primitives.
- **CUDA stream** = a conveyor belt going to a factory floor. You drop tasks on the belt; the factory's own dispatch system runs them. Multiple belts feeding the same factory can run concurrently (subject to factory capacity).
- **CUDA event** = a colored sticker you put on the belt. The factory tells you when the sticker has rolled past, and you can ask how long things took between two stickers.

When ML papers talk about "kernel launch overhead" or "host-bound", they mean time the CPU thread spends submitting work to streams (Python dispatch + libcuda + driver IOCTL). When they talk about "compute-bound", they mean time spent on the GPU's compute units. Different kinds of time, measured with different tools.

**Concepts introduced:** CUDA stream as a work queue (not a thread), asynchronous host/device execution, default-stream pitfalls and per-thread default stream, kernel launch dispatch path, SM (Streaming Multiprocessor), CUDA event, event-based GPU timing, GPU hardware scheduler, host-bound vs compute-bound classification, why CUDA timing requires events (not perf_counter), conveyor-belt mental model, pinned memory + non-blocking transfers, concurrent kernel execution.

---

## Distributed Training

### Q4: What is a GPU rank?

**Date:** 2026-04-24
**Source:** `architecture.md` diagram ("Training Processes (xN) — user-side — one per GPU rank")

**Short answer:** "Rank" is a unique integer ID assigned to each worker process in a distributed training job. If you train on 4 GPUs, you have 4 processes with ranks 0, 1, 2, 3. Each rank owns one GPU and trains a replica of the model on a different data shard. Rank is the universal way distributed code says "which worker am I?"

**Long answer:**

*Where the term comes from.* "Rank" is inherited from **MPI** (Message Passing Interface), the 1990s HPC standard. In MPI, each process in a communicator group has a rank `0..N-1`. PyTorch's `torch.distributed` borrows the term and many of its collective operations straight from MPI.

*Key distributed concepts.*

- **World size**: total number of training processes across the whole job, typically equal to total GPUs. `os.environ["WORLD_SIZE"]`.
- **(Global) rank**: this worker's unique ID, `0..world_size-1`. `os.environ["RANK"]`.
- **Local rank**: this worker's ID *within its node* (machine). `os.environ["LOCAL_RANK"]`.
  - Example: 2 nodes × 4 GPUs each → world_size = 8. On node 0, workers have local_rank 0-3 and global rank 0-3. On node 1, workers have local_rank 0-3 but global rank 4-7.

*How rank is used in training.*

- **GPU binding**: each rank calls `torch.cuda.set_device(local_rank)` so worker 0 uses `cuda:0`, worker 1 uses `cuda:1`, etc. This is convention — the OS doesn't enforce it.
- **Data sharding**: `DistributedSampler(rank=rank, num_replicas=world_size)` gives each worker a non-overlapping slice of the dataset. Worker 0 sees batches 0, N, 2N…; worker 1 sees 1, N+1, 2N+1…; etc.
- **Gradient sync**: `DistributedDataParallel` uses NCCL **all-reduce** after each backward pass — every rank contributes its local gradient, every rank receives the average. After this collective, all N model replicas are in identical state.
- **Logging/checkpointing**: typically only `rank == 0` writes to disk or logs to W&B — otherwise you'd get N duplicate writes. This is why distributed code is littered with `if rank == 0: ...` guards.

*How torchrun sets this up.* `torchrun --nproc_per_node=4 train.py` spawns 4 Python processes. Before running your script, torchrun sets these env vars in each child:

- `RANK`, `LOCAL_RANK`, `WORLD_SIZE`
- `MASTER_ADDR`, `MASTER_PORT` (for **rendezvous** — how workers find each other to form the process group)

Your training code reads these env vars to identify itself and to initialize `torch.distributed`.

*How TraceML uses rank.*

- Each rank detects itself from env vars via [traceml/src/traceml/transport/distributed.py](traceml/src/traceml/transport/distributed.py).
- All telemetry frames are tagged with rank before being shipped over TCP.
- The aggregator's `RemoteDBStore` is *rank-aware* — it keeps rank 0's data separate from rank 1's, etc. That's why the UI can show per-rank panels or aggregate across ranks as needed.

*RL analogy.* This is exactly the identity pattern in A3C / IMPALA: N asynchronous actors, each with a worker ID, contributing to a shared system. PyTorch DDP is more synchronous than IMPALA (all-reduce every step, vs async actor-learner), but the "rank = worker ID" convention is the same.

*OS tangent — does rank 0 really "own" GPU 0?* Not at the OS level. Any process with the right permissions can open any visible CUDA device. `torch.cuda.set_device(local_rank)` just sets a thread-local default for subsequent tensor allocations. You *could* have rank 3 use GPU 0 — DDP simply assumes you won't, because convention is cleaner and NCCL performance depends on a consistent mapping.

**Related files:** [traceml/src/traceml/transport/distributed.py](traceml/src/traceml/transport/distributed.py), [traceml/src/traceml/database/remote_database_store.py](traceml/src/traceml/database/remote_database_store.py)
**Concepts introduced:** rank (MPI heritage), world size, local rank vs global rank, DDP, all-reduce, NCCL, torchrun env vars, `DistributedSampler`, rendezvous, `if rank == 0` idiom

---

### Q12: What is NCCL all-reduce, and why is it a barrier?

**Date:** 2026-04-24
**Source:** suggested follow-up after Q4/Q8 — unlocks all DDP performance discussions

**Short answer:** **NCCL** (NVIDIA Collective Communications Library) is NVIDIA's library for efficient GPU-to-GPU communication — within a node (NVLink, PCIe) and across nodes (Infiniband, RoCE, TCP). **All-reduce** is a collective: every participating rank contributes a tensor and every rank receives the reduced result (typically sum or mean). It's a **barrier** because by definition the operation can't complete until every rank has both contributed *and* received the result — no rank exits until all ranks arrive.

**Long answer:**

*Collective operations (general concept).* A collective involves a group of processes (in PyTorch, the "process group" set up by `init_process_group`) and follows a strict protocol. Common collectives:

- **broadcast(src, tensor)** — src sends, all others receive.
- **reduce(dst, op)** — all ranks contribute, dst receives the reduced result.
- **all-reduce(op)** — all ranks contribute, all ranks receive the reduced result. Equivalent to reduce + broadcast but implemented more efficiently.
- **scatter(src, tensors)** — src splits a tensor, each rank gets a chunk.
- **gather(dst)** — opposite, dst collects all chunks.
- **all-gather** — every rank ends up with all chunks from every other rank.
- **all-to-all** — every rank sends a different chunk to every other rank.

These map directly to MPI primitives, again inheriting from HPC.

*Why all-reduce specifically matters for DDP.* In data-parallel training, every rank holds an identical copy of the model and processes a different mini-batch. After backward, each rank has computed local gradients `g_r` for its mini-batch. To get the average gradient across all ranks (which is what gives correct optimization on the full effective batch), you do:

```
g_global = all_reduce(g_r, op=SUM) / world_size
```

After the all-reduce, every rank has the same `g_global`. Then every rank applies the same optimizer step to its model copy → all replicas stay bit-identical. If you skipped the all-reduce, each rank would update its local model with its local gradient, models would diverge, and you'd no longer have data-parallel training — you'd have N independent training runs on different mini-batches with a different (effectively wrong) loss landscape.

*How NCCL implements all-reduce efficiently.* Naive all-reduce: every rank sends its tensor to a coordinator, coordinator sums, coordinator broadcasts back. O(N) bandwidth at the coordinator → bottleneck.

NCCL uses **ring all-reduce** (Baidu's algorithm, ~2017):

- Arrange the N ranks in a logical ring: 0 → 1 → 2 → ... → N-1 → 0.
- Split each rank's tensor into N chunks.
- **Reduce-scatter phase** (N-1 steps): rank r sends chunk (r-step) to rank (r+1), receives a chunk from rank (r-1), accumulates into its local copy. After N-1 steps, each rank holds the fully-reduced version of one specific chunk.
- **All-gather phase** (N-1 steps): each rank sends its fully-reduced chunk around the ring until everyone has every chunk.
- Total bandwidth per rank: 2(N-1)/N × tensor_size — nearly independent of N. This is why ring all-reduce is the workhorse of large-scale distributed training.

NCCL also picks topology-aware paths: for 8 GPUs in a node connected by NVLink, it builds the ring along NVLink edges. For multi-node, it builds inner rings on each node and an outer ring across nodes. Topology discovery happens during `init_process_group`. Newer NCCL versions support tree-based collectives for very large clusters where rings get too long.

*Why all-reduce is a barrier.* "Barrier" in concurrent programming = a synchronization point that no participant can leave until all participants arrive. All-reduce is a barrier because:

- Rank r can't finish reduce-scatter until it has received a contribution from rank r-1, which can't send until *it* received from r-2, and so on — transitively requires every rank to have contributed.
- Rank r can't finish all-gather until the last chunk arrives via the ring, which transitively requires every rank to have received.
- So no rank exits all-reduce until every rank has both entered *and* progressed through.

Concretely: if rank 3 is slow (slower data loading, slower backward, longer GPU kernel), ranks 0, 1, 2 will block at the all-reduce call waiting for rank 3 to participate. This is the **straggler effect** referenced in Q8.

*Sync vs async all-reduce in PyTorch.* Default `DistributedDataParallel` overlaps backward computation with all-reduce: as gradients become ready in backward (last layer's gradients are ready first), they're chunked into "buckets" and the all-reduce on each bucket starts immediately, while earlier layers' backward continues. This hides communication latency behind computation, but the final synchronization point is still a barrier — the optimizer step can't run until all bucket all-reduces have completed.

For huge models sharded with FSDP/ZeRO, there's also all-gather (for parameters before forward) and reduce-scatter (for gradients during backward). Same barrier semantics.

*Why TraceML cares.*

- TraceML's overhead has to be small relative to the all-reduce cost, or it would become the straggler. Because telemetry sends are off the critical path (asynchronous TCP send to aggregator, no rank waits for a reply), TraceML adds ~zero barrier-side latency.
- Bottleneck diagnostics in TraceML often surface as "rank N spends X% of its time blocked in NCCL barrier" — that's a sign that other ranks finished earlier and are waiting on N. The interesting question becomes: why is N slow? Dataloader stall? Slow GPU? Different mini-batch composition? This is exactly why per-rank visibility (`RemoteDBStore` keying telemetry by rank) is essential.
- Future product features around DDP straggler analysis would lean directly on this: comparing per-rank step times, identifying the rank that consistently arrives last, surfacing the cause.

*Concrete numbers (back-of-envelope, mid-2020s hardware).* For an 8-GPU DGX node with a 350 MB gradient tensor (mid-size transformer):

- NVLink (~600 GB/s effective): ~0.6 ms ring all-reduce.
- PCIe Gen4 (~25 GB/s): ~25 ms.
- 32 GPUs across 4 nodes via Infiniband 200 Gb/s: ~1–2 ms intra-node + ~10–15 ms inter-node hop. Total: ~15–20 ms.

This is on the same order as a forward+backward at small/medium batch sizes, which is why communication efficiency dominates large-scale training.

**Concepts introduced:** NCCL, collective operations (broadcast, reduce, all-reduce, scatter, gather, all-gather, all-to-all), MPI heritage of collectives, ring all-reduce algorithm (reduce-scatter + all-gather phases), bandwidth-optimal collective, topology-aware ring construction, DDP gradient sync, sync barrier semantics, DDP gradient bucketing / overlap with backward, FSDP/ZeRO collectives, straggler effect, NVLink vs PCIe vs Infiniband bandwidth tiers.

---

## PyTorch (separate file)

PyTorch fundamentals through internals live in a companion file: [traceml_pytorch_qa.md](traceml_pytorch_qa.md). It uses **P1, P2, …** numbering and currently has **P1–P52** covering:

- Tensor fundamentals (P1–P6)
- nn.Module mechanics (P7–P12)
- Autograd (P13–P19)
- Optimizers (P20–P23)
- DataLoader (P24–P29)
- CUDA dispatch path & memory model (P30–P34)
- Mixed precision / AMP (P35–P37)
- Distributed (DDP / FSDP) (P38–P42)
- Eager vs graph mode (`torch.compile`, JIT, fx) (P43–P45)
- Checkpointing & state (P46–P47)
- PyTorch internals relevant to TraceML (P48–P52)

Cross-references between the files: a P-entry in the PyTorch file links back to Q-entries here; new questions there get the next P-number. Walkthroughs of TraceML code live in [traceml_learning_code_walkthroughs.md](traceml_learning_code_walkthroughs.md) (W-numbers).

---

## TraceML Architecture

### Q2: What is "server-side" vs "user-side"?

**Date:** 2026-04-24
**Source:** `architecture.md` diagram labels ("server-side, long-running" on aggregator; "user-side — one per GPU rank" on training processes)

**Short answer:** It's the classic **client-server split**. **User-side** = the training processes, where *your* PyTorch code runs (train.py, model, dataloader, optimizer). TraceML injects its hooks in-process alongside your code. **Server-side** = the aggregator, a separate Python process that owns no user code. Its only job: listen on a TCP port, receive telemetry from all ranks, store it, render the UI. All N training processes are clients of the one aggregator server.

**Long answer:**

*The client-server pattern.* One party (the **server**) waits on a known address (here, a TCP port on localhost). One or more parties (**clients**) connect and push requests. In TraceML the "requests" are telemetry rows (step timings, memory samples, system samples). The aggregator accepts connections from all N training ranks and stores everything keyed by rank.

*Why this split matters.*

1. **Fail-open**: the aggregator runs completely separate code paths (Rich rendering, NiceGUI, SQL). If any of that breaks — crash of the renderer, port conflict, NiceGUI bug — the aggregator dies but your training continues. Worst case: no telemetry. You never lose training progress to a monitoring bug. This is explicitly called out in the diagram: *"Fail-open: aggregator crash never crashes training."*
2. **Overhead budget**: forward/backward has to be microsecond-cheap. Rich rendering a table, running SQL, or pushing a NiceGUI websocket frame costs milliseconds. Offloading all that to a separate process keeps the cost out of the hot path.
3. **Symmetry of ranks**: if the aggregator were a thread inside rank 0, that rank would do more work than the others — asymmetry that corrupts the very telemetry you're trying to measure. External aggregator keeps all ranks peers.
4. **Scalability**: once it's TCP-based, nothing stops ranks on a different machine from connecting to a central aggregator.

*Why the term "user-side"?* Two parties own code in the same training process: **you** wrote train.py; **TraceML** attaches hooks. The term "user-side" emphasizes that the training process is *your* turf with TraceML as a guest. The aggregator, by contrast, is 100% TraceML — zero user code runs there.

*RL analogy.* This is the same separation as IMPALA / Ape-X — distributed **actors** (user-side) push experience to a central **learner** (server-side). The actors don't care about storage or policy updates; the learner doesn't own environment interaction. Clean division of responsibility — and you already know why that design choice beats monolithic: decoupled failure modes, decoupled scaling, decoupled overhead budgets.

**Related files:** [traceml/src/traceml/runtime/](traceml/src/traceml/runtime/) (user-side agent), [traceml/src/traceml/aggregator/](traceml/src/traceml/aggregator/) (server-side)
**Concepts introduced:** client-server pattern, in-process vs out-of-process, fail-open, overhead budget, rank symmetry

---

### Q3: What does "long-running" mean?

**Date:** 2026-04-24
**Source:** `architecture.md` diagram ("server-side, long-running" on aggregator)

**Short answer:** A long-running process stays alive for the entire duration of a job — minutes, hours, or days — rather than starting, doing one small thing, and exiting. The aggregator boots when training starts and only exits when training ends. Contrast with short-lived processes like `ls` (10 ms) or a single HTTP request handler (100 ms).

**Long answer:**

*What "long-running" implies for the design.*

- **Persistent state**: the aggregator holds all telemetry in RAM (in deque-based tables at [traceml/src/traceml/database/database.py](traceml/src/traceml/database/database.py)). A short-lived process wouldn't need this — it would handle one request and discard state.
- **Main event loop**: a long-running server has a forever-loop — "accept connection, read frame, store, repeat." It doesn't terminate on its own; it terminates when told to.
- **Bounded memory**: if samples arrive at 1 Hz and you train for 7 days, you get ~600K samples per sampler per rank. Unbounded storage = OOM. TraceML's deques have a `maxlen`, so old rows evict automatically. This design is a *direct consequence* of being long-running.
- **Signal handling**: long-running processes must respond cleanly to `SIGTERM` (Ctrl-C), `SIGHUP`, etc. They can't just "exit whenever" — they need to flush buffers, close sockets, write final summaries.
- **Resource hygiene**: over days, file-descriptor leaks, memory fragmentation, and log growth accumulate. Short-lived processes dodge these by dying before anything leaks enough to matter.


|                 | Short-lived             | Long-running                             |
| --------------- | ----------------------- | ---------------------------------------- |
| Examples        | `ls`, `cat`, a cron run | training job, aggregator, web server, DB |
| State lifetime  | Dies on exit (fine)     | Must be bounded or persisted             |
| Signal handling | Just die                | Flush, close, checkpoint                 |
| Leaks           | Don't matter            | Kill you eventually                      |


*Why the aggregator specifically is long-running:* it must observe the full training job end-to-end. A 7-day pretraining run needs continuous telemetry. If the aggregator restarted periodically, you'd get gaps in per-run views.

**Related files:** [traceml/src/traceml/database/database.py](traceml/src/traceml/database/database.py) (bounded deques), [traceml/src/traceml/aggregator/](traceml/src/traceml/aggregator/) (event loop)
**Concepts introduced:** long-running vs short-lived, event loop, bounded buffers, signal handling, `maxlen` deque

---

### Q8: Why TraceML uses subprocesses (expanded)

**Date:** 2026-04-24
**Source:** deeper expansion of Q1's "Why TraceML uses subprocesses (not threads)"

**Short answer:** Four reinforcing reasons. **(1) Crash isolation** — aggregator dies, training survives, because processes don't share address space. **(2) GIL avoidance** — aggregator's Python work doesn't steal bytecode time from training. **(3) Rank symmetry** — no one rank carries extra work; DDP all-reduce isn't throttled by a straggler. **(4) torchrun composability** — PyTorch's launcher is subprocess-based, so a subprocess architecture composes without fighting the launcher.

**Long answer:**

*

1. Crash isolation.* Threads share the process's virtual address space. One bad pointer in a C extension takes down the whole process. One uncaught assertion in NiceGUI or a msgpack decoder kills the Python interpreter for everyone in that process. Processes don't share address space. A crash in the aggregator only terminates the aggregator; the kernel cleans it up. The training process doesn't even notice, except that its next TCP send fails. TraceML's runtime handles that: failed sends are logged, training continues (fail-open). This is the **most important reason**. The product-level contract — *"I can leave TraceML running for 7 days and it won't hurt my training"* — hinges entirely on this.

*

1. GIL avoidance.* From Q6: within one process, only one thread can execute Python bytecode at a time. If the aggregator ran as a thread in a training process, its Python work (parsing telemetry frames, updating tables, driving renderers) would steal GIL time from the training loop. On a fast iteration (~50 ms/step), even a few ms of GIL contention is meaningful slowdown. Moving the aggregator to a separate process means zero GIL contention between the two. The only shared cost is kernel time for TCP send/recv — nanoseconds on localhost. Cheap.

*

1. Rank symmetry.* In a DDP job with N ranks, each rank runs the same training loop, each hits `DistributedDataParallel`'s backward, each participates in NCCL all-reduce. All-reduce is a **synchronization barrier**: it can only finish when the slowest rank arrives. If one rank is doing extra work (hosting an aggregator thread), it's a **straggler** — everyone else waits on it. Effective throughput drops to the straggler's speed. An external aggregator process keeps every rank identical; no one carries extra load. The architecture diagram's line *"DDP fan-in: N ranks all feed a single aggregator"* captures this: the aggregator is off to the side, not embedded in any rank.

*

1. torchrun composability.* `torchrun` expects to be the launcher: it runs `python train.py` N times with rank env vars set. If TraceML tried to run the aggregator as a thread inside train.py, you'd spin up N aggregators (one per rank) — all fighting for the same TCP port, all rendering the same UI. By having the CLI spawn the aggregator *separately* and then invoke torchrun, TraceML gets exactly one aggregator regardless of rank count. Clean composition. This also makes the multi-node story possible: rank 4 on node 1 can be configured to push telemetry to rank 0's host.

*Counterargument: why not `multiprocessing.Process` from within train.py?* You could imagine an alternative where train.py itself forks an aggregator process via `multiprocessing`. That would also give process isolation. Problems:

- It inverts the launch order: every integration (HF, Lightning, Ignite, Pax, ...) would need to know to start the aggregator. Putting the spawn in the CLI keeps integrations thin.
- Multi-node: if train.py is running on node 1, how does it find the aggregator on node 0? With a top-level CLI, env vars pointing to rank 0's host configure this cleanly.
- CUDA fork-safety: if train.py has already called `torch.cuda.`*, forking from there is dangerous.

**Concepts introduced:** crash isolation via address-space separation, GIL contention in the same process, NCCL all-reduce barrier, straggler effect in DDP, launcher composability, CUDA fork-safety.

---

