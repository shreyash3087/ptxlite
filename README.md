# ptxlite

A small C++17 compiler from a CUDA-style kernel language to PTX.

---

## Architecture

```
 Source (.k)
     │
     ▼
 ┌─────────┐   tokens   ┌─────────┐    AST    ┌──────┐
 │  Lexer  │──────────▶│  Parser │──────────▶│ Sema │
 └─────────┘            └─────────┘           └──────┘
                                                  │ typed AST
                                                  ▼
                                            ┌─────────┐
                                            │  Lower  │  (AST → IR)
                                            └─────────┘
                                                  │ IR (basic blocks)
                                                  ▼
                                            ┌─────────┐
                                            │   Opt   │  (fold, DCE, FMA)
                                            └─────────┘
                                                  │ optimized IR
                                          ┌───────┴───────┐
                                          ▼               ▼
                                     ┌─────────┐    ┌─────────┐
                                     │   PTX   │    │  Interp │
                                     │ Emitter │    │(testing)│
                                     └─────────┘    └─────────┘
```

---

## The language

```c
kernel saxpy(float* x, float* y, float a, int n) {
  int i = gid();
  if (i < n) { y[i] = a * x[i] + y[i]; }
}
```

### Types
- `int`, `float` — 32-bit scalars.  
- `int*`, `float*` — pointer parameters (no local pointer variables).

### Builtins
| Function | Description |
|---|---|
| `gid()` | `blockIdx() * blockDim() + threadIdx()` |
| `threadIdx()` | thread index within block |
| `blockIdx()` | block index |
| `blockDim()` | block dimension |
| `sqrt(x)` | `float` square root |
| `min(a, b)` / `max(a, b)` | numeric min/max |

### Control flow
`if`/`else`, `while`, `for` (desugared to `while`), `return`.

### Operators
`+ - * / %`, comparisons, `&& || !` (short-circuit), `+= -= *= /=`.  
Explicit casts: `int(...)`, `float(...)`.  
Uninitialized locals are zero-initialized.

---

## Optimization passes (src/opt.hpp)

| Pass | What it does |
|---|---|
| Constant folding | Folds `2 + 3` → `5`, constant comparisons |
| Copy propagation | Eliminates redundant moves |
| Algebraic identities | `x*1→x`, `x-x→0`, `x*0→0`, etc. |
| Strength reduction | `x * 2^k` → `shl.b32 x, k` |
| Branch folding | Removes unreachable branches |
| Unreachable block removal | Prunes dead basic blocks |
| Block merging | Fuses straight-line blocks |
| DCE | Dead-code elimination (iterated) |
| Mul-add fusion | `a*b + c` → `fma.rn.f32` / `mad.lo.s32` at `-O2` |

---

## Build

Requires a C++17 compiler (GCC ≥ 9 or Clang ≥ 10) and GNU Make.

```bash
make          # builds build/ptxlite and build/test_main
make test     # runs the test suite
make clean
```

**Windows (MinGW/MSYS2):**

```bash
g++ -std=c++17 -Wall -Wextra -O2 -ffp-contract=off -Isrc -o ptxlite src/main.cpp
g++ -std=c++17 -Wall -Wextra -O2 -ffp-contract=off -Isrc -o test_main tests/test_main.cpp
```

---

## CLI usage

```
ptxlite [options] <input.k>

Options:
  -O0 / -O1 / -O2      optimization level (default -O2)
  --no-fma              disable mul-add fusion
  --arch=sm_XX          PTX target (default sm_75)
  --emit-ir             print IR to stdout and exit
  --emit-ptx            print PTX to stdout (default when -o is absent)
  -o <file>             write PTX to file
  --stats               print optimization statistics to stderr
```

Example:

```bash
./build/ptxlite -O2 --stats examples/saxpy.k -o saxpy.ptx
```

---

## PTX validation with ptxas

You can validate the emitted PTX without a GPU:

```bash
# download ptxas from the CUDA nvcc wheel (no GPU needed)
pip download nvidia-cuda-nvcc-cu12 --no-deps -d tools
cd tools && unzip -o *.whl 'nvidia/cuda_nvcc/bin/ptxas'
./nvidia/cuda_nvcc/bin/ptxas -arch=sm_75 saxpy.ptx -o /dev/null
```

To check all examples at once:

```bash
chmod +x scripts/check_ptx.sh
./scripts/check_ptx.sh ./tools/nvidia/cuda_nvcc/bin/ptxas
```

---

## Examples

| File | Description |
|---|---|
| `examples/saxpy.k` | y[i] = a·x[i] + y[i] |
| `examples/rowsum.k` | Row sum of a matrix |
| `examples/collatz.k` | Collatz step count |
| `examples/mandelbrot.k` | Mandelbrot iteration count |
| `examples/mathops.k` | sqrt, min, max showcase |

---

## Tests

The test suite (`tests/test_main.cpp`) covers:

- **Correctness** at O0, O1, O2: saxpy, rowsum, collatz, mandelbrot, compared against host reference implementations.
- **Short-circuit safety**: launching more threads than `n` must not cause an out-of-bounds store.
- **Optimizer checks**: `fma.rn.f32` and `mad.lo.s32` present at O2; absent with `--no-fma`; `shl.b32` for power-of-2 multiplies; constant folding.
- **Diagnostics**: line numbers for type errors and undeclared-identifier errors.
- **Fuzzer**: 200 randomly generated integer-arithmetic programs, O0 vs O1 vs O2 differential testing.

---

## Known limitations

- No user-defined functions (only built-ins).
- No shared memory, atomics, or warp intrinsics.
- Only 1-D thread indexing (`gid()`).
- PTX is `sm_75` / PTX 7.0 by default; change with `--arch`.
- Division by zero yields 0 (matches PTX undefined behavior).
- Float contraction at `-O2` changes rounding vs O0; compare with `--no-fma` for exact float tests.
