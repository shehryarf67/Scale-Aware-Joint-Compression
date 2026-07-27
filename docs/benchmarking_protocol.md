# Benchmarking protocol

## The rule

**All final deployment measurements in this study run on the same CPU, with a fixed thread count, a
fixed sequence length, and a fixed batch size, after warm-up, over repeated runs, reported as median
and p95.**

Latency, throughput, peak memory, and checkpoint size are deployment measurements. So is the final
quality evaluation. Nothing that appears in a results table is measured on GPU.

## What GPUs may be used for

| Allowed on GPU                      | Must be CPU                            |
| ----------------------------------- | -------------------------------------- |
| recovery fine-tuning                | latency (mean, median, p95, std)       |
| joint compression training          | throughput (tokens/s)                  |
| quantisation calibration            | peak process memory                    |
| exploratory quality evaluation      | checkpoint size                        |
| loss curves during training         | **final reported** quality evaluation  |

The split follows one principle: a GPU may be used to *produce* a compressed model, never to
*measure* one.

## Why CPU

1. **It is the question.** Secondary question 4 asks whether theoretical sparsity produces real CPU
   latency improvements. Only a CPU measurement answers that.
2. **It is where these models deploy.** A 160M–1.4B compressed model is deployed on commodity CPU far
   more often than on a datacentre GPU. Compression matters most exactly where there is no
   accelerator.
3. **GPU latency hides the effect.** At batch 1 and short sequences a GPU is launch-latency bound, so
   removing arithmetic changes almost nothing and every arm measures the same.
4. **Quantised INT8 kernel support is well established on CPU.** PyTorch's INT8 CPU backends are
   mature, so a CPU measurement of an INT8 model is measuring a deployment path people actually use.
   This does *not* extend to every bit width — see the backend section below.

## Backend and bit-width constraints

**Read this before running the aggressive budget.** The measurement protocol below is sound only if
every row being compared was produced by the same runtime on the same artefact format.

- **PyTorch's native CPU quantisation support is strongest for INT8.** That is the well-trodden path,
  and it is what the moderate budget uses.
- **4-bit weight-only CPU deployment may require a separate backend** — a packed-weight custom linear
  module, or an external runtime. There is no equally mature built-in 4-bit CPU kernel.
- **Latency and size results are not comparable if the moderate and aggressive settings use different
  runtimes or artefact formats.** Each budget may be internally valid while the comparison *between*
  them measures the runtime rather than the compression. The same applies with more force across arms:
  a 4-bit joint artefact benchmarked against an INT8 sequential artefact is not a joint-gain
  measurement.
- **The final backend decision must be made before the main experiments begin**, not after seeing which
  choice produces a more favourable result. It constrains which bit widths the study can use at all.

Documented fallback, if a single 4-bit CPU path cannot be implemented for both the sequential and joint
arms:

| Budget | Sparsity | Bit width |
| --- | --- | --- |
| Moderate | 50% | INT8 |
| Aggressive (fallback) | 70% | INT8 |

This keeps every row on one runtime and one artefact format, at the cost of making precision a constant
rather than a second compression axis. 4-bit support remains in the configuration system either way;
what the fallback changes is whether the *main study* uses it. See
[method_definition.md](method_definition.md#bit-widths-and-the-4-bit-risk) and the commented fallback
block in [`main_scale_sweep.yaml`](../configs/experiments/main_scale_sweep.yaml).

Whatever is chosen, record `quantisation.backend` and the artefact format in every run record, and
check they agree before putting two rows in the same table.

## Fixed parameters

| Parameter                  | Value    | Why |
| -------------------------- | -------- | --- |
| `device`                   | `cpu`    | Not overridable; the config loader rejects anything else. |
| `num_threads`              | 4        | Realistic commodity deployment, small enough that thread scheduling does not dominate. |
| `interop_threads`          | unset    | PyTorch allows this to be set only once per process, before any parallel work. |
| `batch_size`               | 1        | Single-request serving, the latency-sensitive case. |
| `sequence_length`          | 128      | Short enough to measure per-request latency rather than throughput. |
| `generated_tokens`         | 0        | Single forward pass (prefill). Set >0 to measure decoding instead. |
| `warmup_runs`              | 5        | Untimed. See below. |
| `measured_runs`            | 30       | Enough for a stable median and a meaningful p95. |
| `fail_on_thread_mismatch`  | `true`   | A silently unpinned run produces plausible, incomparable numbers. |

Defined in [cpu_benchmark.yaml](../configs/evaluation/cpu_benchmark.yaml).

## Procedure

```
1. pin threads          torch.set_num_threads(num_threads) and set OMP/MKL/OPENBLAS/NUMEXPR env vars
2. verify the pin       compare torch.get_num_threads() against the request; fail on mismatch
3. build the input      one fixed synthetic batch, allocated once, outside the timed region
4. record baseline RSS  before any warm-up
5. warm up              warmup_runs untimed iterations
6. measure              measured_runs timed iterations, each recorded individually
7. sample peak RSS
8. summarise            mean, median, std, p95, p99, min, max
9. derive throughput    from the MEDIAN latency, not the mean
10. capture metadata    full hardware and software details
11. record              JSON with per-run latencies plus one flat CSV row
```

Implemented in [cpu.py](../src/scale_aware_compression/benchmarking/cpu.py). The runner times an
arbitrary zero-argument callable, so the protocol is identical across all five arms and is
unit-testable without a model.

## Rules with reasons

### Threads must be pinned, and the pin verified

BLAS libraries read `OMP_NUM_THREADS` at first use, so setting `torch.set_num_threads` alone is not
always enough. Both are set, and `torch.get_num_threads()` is checked afterwards. A mismatch fails
the run by default: latencies measured under different thread counts are not comparable, and nothing
about the resulting number looks wrong.

### Warm-up is not optional

The first call to a quantised CPU kernel is often several times slower than the steady state —
kernel selection, allocator growth, cold caches, and lazy initialisation all land on it. With
`warmup_runs: 0` a 30-run measurement is dominated by one-off cost, which inflates the mean and the
p95 unevenly across arms.

### Report median and p95, not just the mean

The mean is sensitive to a single scheduling hiccup. The median is the central measure this protocol
reports, and p95 is the tail a deployment actually experiences. Mean and standard deviation are also
recorded, so the distribution can be inspected.

Throughput is derived from the **median**, so it stays consistent with the reported latency.

### Percentiles use linear interpolation

Matching `numpy.percentile`'s default, implemented in
[latency.py](../src/scale_aware_compression/benchmarking/latency.py) without a NumPy dependency. With
30 samples, p95 falls between order statistics and the interpolation choice matters, so it is stated
rather than left implicit.

### Watch the coefficient of variation

The runner warns when std/mean exceeds 15%. That usually means background load. **Re-run rather than
report:** a noisy measurement of the joint arm against a clean one of the sequential arm can produce
a joint gain from nothing.

### One machine per results table

CPU latencies from different machines are not comparable and must never be averaged. Every record
stores `cpu_model`, core counts, memory, thread environment, and the resolved torch version. Check
that a table's rows agree on hardware before reading it.

### Idle machine

Close everything else. No compiles, no browser, no background sync. On a laptop, plug in and disable
CPU frequency scaling if the platform allows it — thermal throttling part-way through a 30-run
measurement shows up as a bimodal latency distribution.

### Size means weight files only

Tokeniser and config JSON are byte-identical across arms, so including them adds a constant that
shrinks the apparent compression ratio of smaller models more than larger ones. Only
`.safetensors`, `.bin`, `.pt`, `.pth`, `.gguf`, and `.onnx` are counted.

`storage_efficiency` compares the measured size against what the budget implies. Well below 1.0
means sparsity or low precision was not realised in the serialised format — usually a conversion
step that silently no-oped.

### Peak memory is process RSS

Not a torch allocator statistic. The deployment question is how much memory the serving process
needs, and quantised CPU models keep packed weights outside the caching allocator, so an
allocator-based figure would understate them. Linux and Windows expose a true high-water mark; macOS
falls back to current RSS, which under-reports a peak already released.

## Reporting the theoretical bound alongside

Every latency row is reported next to `1 / (1 - sparsity)`, the speedup a kernel that skipped every
zero at no overhead would give. `sparsity_realisation` is the fraction of that bound actually
achieved:

```
sparsity_realisation = (measured_speedup - 1) / (theoretical_speedup - 1)
```

1.0 means fully realised; 0.0 means none of it. **A near-zero realisation for unstructured sparsity
is the expected result and a finding worth reporting**, not a bug: a dense GEMM kernel performs the
same multiply-accumulates whether or not the operands are zero, so the zeros are multiplied rather
than skipped. Realising a gain from sparsity requires a kernel that exploits the pattern, which is a
property of the runtime and not of the compression method.

The 2:4 semi-structured variant exists to test whether a pattern *more likely* to admit such a kernel
does better. Verify that the installed CPU backend actually provides a 2:4 kernel before reading its
row — support for semi-structured sparsity on CPU is much less established than the GPU equivalent, and
without a kernel the row measures a dense operation on a patterned weight matrix. Record which kernel
was used.

## Optional: thread-count sweep

Sparsity benefits can scale differently with parallelism than with arithmetic intensity. Run the
same model at several thread counts:

```bash
for threads in 1 2 4 8; do
  python scripts/run_cpu_benchmark.py \
    --config configs/experiments/main_scale_sweep.yaml \
    --threads "$threads"
done
```

Report each thread count as its own series. **Never average across thread counts.**

## Verification before reporting

- [ ] All rows share one `hardware_cpu_model`
- [ ] All rows share one `benchmark_num_threads`, `batch_size`, and `sequence_length`
- [ ] `thread_report.torch_num_threads` equals the requested count on every row
- [ ] Coefficient of variation under 15% everywhere
- [ ] `warmup_runs` ≥ 5 and `measured_runs` ≥ 30
- [ ] Throughput consistent with the median latency and the token count
- [ ] `storage_efficiency` plausible for every quantised artefact
- [ ] `is_converted` true for every quantised and joint artefact
- [ ] Theoretical bound reported next to every measured speedup
- [ ] All rows share one `quantisation.backend` and one artefact format
- [ ] Sequential and joint rows for a given budget used the same runtime
- [ ] If moderate and aggressive rows used different runtimes, they are reported as separate,
      non-comparable measurement sets rather than as one trade-off curve
- [ ] For a 2:4 row, the kernel actually used is recorded (sparse kernel or dense fallback)
