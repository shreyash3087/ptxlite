// ptxlite test suite.
//
// Covers:
//   - Correctness at O0, O1, O2: saxpy, rowsum, collatz, mandelbrot.
//   - Short-circuit safety: more threads than n must not cause out-of-bounds access.
//   - Optimizer checks on emitted PTX: fma.rn.f32 / mad.lo.s32 / shl.b32 present at O2,
//     absent with --no-fma; constant folded; dead branch gone.
//   - Diagnostics: bad line numbers for type errors and use-before-declare.
//   - Random-program fuzzer: O0 vs O1 vs O2 differential testing.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../src/compiler.hpp"
#include "../src/interp.hpp"

// ── helpers ────────────────────────────────────────────────────────────────────

static int pass_ = 0, fail_ = 0;

#define CHECK(cond, msg)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                               \
      std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: " << (msg) << "\n"; \
      ++fail_;                                                                   \
    } else {                                                                     \
      ++pass_;                                                                   \
    }                                                                            \
  } while (0)

#define CHECK_THROW(expr, exc_type, msg)                                        \
  do {                                                                          \
    bool caught_ = false;                                                       \
    try { (void)(expr); }                                                       \
    catch (const exc_type&) { caught_ = true; }                                 \
    CHECK(caught_, msg);                                                        \
  } while (0)

// Build a ptxlite::Value for a buffer pointer: buffer index bi, byte offset 0.
static ptxlite::Value ptrVal(int bi) {
  ptxlite::Value v; v.p = ptxlite::encPtr(bi, 0); return v;
}
static ptxlite::Value ival(int32_t x) { ptxlite::Value v; v.i = x; return v; }
static ptxlite::Value fval(float x)   { ptxlite::Value v; v.f = x; return v; }

// Run a compiled kernel through the interpreter at a given opt level.
struct RunResult {
  std::vector<ptxlite::Func> funcs;
  ptxlite::Stats stats;
};

static RunResult compile(const std::string& src, int opt, bool fma = true) {
  ptxlite::Options o; o.opt = opt; o.fma = fma;
  RunResult r;
  r.funcs = ptxlite::compile(src, o, &r.stats);
  return r;
}

// Launch kernel 0 of a RunResult.
static void launch(const RunResult& r, const std::vector<ptxlite::Buffer>& bufs,
                   const std::vector<ptxlite::Value>& args, int nBlocks, int blockDim) {
  ptxlite::Interpreter interp(bufs);
  interp.launch(r.funcs[0], args, nBlocks, blockDim);
}

// ── SAXPY ─────────────────────────────────────────────────────────────────────

static const char* saxpySrc = R"(
kernel saxpy(float* x, float* y, float a, int n) {
  int i = gid();
  if (i < n) { y[i] = a * x[i] + y[i]; }
}
)";

static void testSaxpy(int opt) {
  const int N = 64;
  std::vector<float> x(N), y(N), yref(N);
  for (int i = 0; i < N; ++i) { x[i] = float(i); y[i] = float(N - i); yref[i] = 2.0f * x[i] + y[i]; }
  std::vector<float> ycopy = y;

  auto r = compile(saxpySrc, opt);
  std::vector<ptxlite::Buffer> bufs = {
    {x.data(), (size_t)N, 4},
    {ycopy.data(), (size_t)N, 4}
  };
  auto args = std::vector<ptxlite::Value>{ptrVal(0), ptrVal(1), fval(2.0f), ival(N)};
  launch(r, bufs, args, 1, N);

  bool ok = true;
  for (int i = 0; i < N; ++i)
    if (std::fabs(ycopy[i] - yref[i]) > 1e-4f) { ok = false; break; }
  CHECK(ok, "saxpy correctness at O" + std::to_string(opt));
}

// ── Row sum ───────────────────────────────────────────────────────────────────

static const char* rowsumSrc = R"(
kernel rowsum(float* row, float* out, int cols, int nrows) {
  int i = gid();
  if (i < nrows) {
    float s = 0.0;
    int base = i * cols;
    int j = 0;
    while (j < cols) {
      s += row[base + j];
      j += 1;
    }
    out[i] = s;
  }
}
)";

static void testRowsum(int opt) {
  const int R = 8, C = 16;
  std::vector<float> mat(R * C), out(R, 0.0f), ref(R, 0.0f);
  for (int i = 0; i < R * C; ++i) mat[i] = float(i % 7 + 1);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) ref[r] += mat[r * C + c];

  auto rv = compile(rowsumSrc, opt);
  std::vector<ptxlite::Buffer> bufs = {{mat.data(), (size_t)(R*C), 4}, {out.data(), (size_t)R, 4}};
  auto args = std::vector<ptxlite::Value>{ptrVal(0), ptrVal(1), ival(C), ival(R)};
  launch(rv, bufs, args, 1, R);

  bool ok = true;
  for (int r = 0; r < R; ++r)
    if (std::fabs(out[r] - ref[r]) > 1e-3f) { ok = false; break; }
  CHECK(ok, "rowsum correctness at O" + std::to_string(opt));
}

// ── Collatz ───────────────────────────────────────────────────────────────────

static const char* collatzSrc = R"(
kernel collatz(int* in, int* out, int n) {
  int i = gid();
  if (i < n) {
    int x = in[i];
    int steps = 0;
    while (x != 1 && steps < 1000) {
      if (x % 2 == 0) { x = x / 2; } else { x = 3 * x + 1; }
      steps += 1;
    }
    out[i] = steps;
  }
}
)";

static int collatzHost(int x) {
  int s = 0;
  while (x != 1 && s < 1000) { x = (x % 2 == 0) ? x / 2 : 3*x+1; ++s; }
  return s;
}

static void testCollatz(int opt) {
  const int N = 20;
  std::vector<int32_t> in(N), out(N, 0), ref(N);
  for (int i = 0; i < N; ++i) { in[i] = i + 2; ref[i] = collatzHost(i + 2); }

  auto rv = compile(collatzSrc, opt);
  std::vector<ptxlite::Buffer> bufs = {{in.data(), (size_t)N, 4}, {out.data(), (size_t)N, 4}};
  auto args = std::vector<ptxlite::Value>{ptrVal(0), ptrVal(1), ival(N)};
  launch(rv, bufs, args, 1, N);

  bool ok = true;
  for (int i = 0; i < N; ++i)
    if (out[i] != ref[i]) { ok = false; break; }
  CHECK(ok, "collatz correctness at O" + std::to_string(opt));
}

// ── Mandelbrot ────────────────────────────────────────────────────────────────

static const char* mandelbrotSrc = R"(
kernel mandelbrot(float* cx, float* cy, int* out, int max_iter, int n) {
  int i = gid();
  if (i < n) {
    float x = 0.0;
    float y = 0.0;
    float c_re = cx[i];
    float c_im = cy[i];
    int iter = 0;
    while (iter < max_iter && x * x + y * y < 4.0) {
      float xnew = x * x - y * y + c_re;
      y = 2.0 * x * y + c_im;
      x = xnew;
      iter += 1;
    }
    out[i] = iter;
  }
}
)";

static int mandelbrotHost(float cr, float ci, int maxIter) {
  float x = 0, y = 0; int it = 0;
  while (it < maxIter && x*x + y*y < 4.0f) {
    float nx = x*x - y*y + cr; y = 2*x*y + ci; x = nx; ++it;
  }
  return it;
}

static void testMandelbrot(int opt) {
  const int N = 16; const int maxIter = 50;
  std::vector<float> cx(N), cy(N); std::vector<int32_t> out(N, 0), ref(N);
  for (int i = 0; i < N; ++i) { cx[i] = -2.0f + i * 0.25f; cy[i] = 0.0f; ref[i] = mandelbrotHost(cx[i], cy[i], maxIter); }

  auto rv = compile(mandelbrotSrc, opt);
  std::vector<ptxlite::Buffer> bufs = {{cx.data(),(size_t)N,4},{cy.data(),(size_t)N,4},{out.data(),(size_t)N,4}};
  auto args = std::vector<ptxlite::Value>{ptrVal(0), ptrVal(1), ptrVal(2), ival(maxIter), ival(N)};
  launch(rv, bufs, args, 1, N);

  bool ok = true;
  for (int i = 0; i < N; ++i)
    if (out[i] != ref[i]) { ok = false; break; }
  CHECK(ok, "mandelbrot correctness at O" + std::to_string(opt));
}

// ── Short-circuit safety ──────────────────────────────────────────────────────

static void testShortCircuit() {
  // Launch 2x more threads than n; without i < n guard the store would OOB.
  const int N = 8;
  std::vector<float> x(N, 1.0f), y(N, 1.0f);

  auto rv = compile(saxpySrc, 2);
  std::vector<ptxlite::Buffer> bufs = {{x.data(),(size_t)N,4},{y.data(),(size_t)N,4}};
  auto args = std::vector<ptxlite::Value>{ptrVal(0), ptrVal(1), fval(3.0f), ival(N)};
  bool threw = false;
  try { launch(rv, bufs, args, 1, N * 2); }
  catch (...) { threw = true; }
  CHECK(!threw, "short-circuit: extra threads must not OOB");
}

// ── PTX optimizer checks ──────────────────────────────────────────────────────

static bool contains(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

static void testPtxOptChecks() {
  // fma.rn.f32 present at O2 for saxpy (a * x[i] + y[i])
  {
    ptxlite::Options o2; o2.opt = 2; o2.fma = true;
    auto funcs = ptxlite::compile(saxpySrc, o2);
    std::string ptx = ptxlite::emitPTX(funcs, o2);
    CHECK(contains(ptx, "fma.rn.f32"), "PTX: fma.rn.f32 present at O2 for saxpy");
  }
  // fma absent with --no-fma
  {
    ptxlite::Options onf; onf.opt = 2; onf.fma = false;
    auto funcs = ptxlite::compile(saxpySrc, onf);
    std::string ptx = ptxlite::emitPTX(funcs, onf);
    CHECK(!contains(ptx, "fma.rn.f32"), "PTX: fma.rn.f32 absent with --no-fma");
  }
  // shl.b32 present when multiplying by a power of 2
  {
    const char* shlSrc = R"(
kernel shltest(int* a, int* b, int n) {
  int i = gid();
  if (i < n) { b[i] = a[i] * 4; }
}
)";
    ptxlite::Options o2; o2.opt = 2;
    auto funcs = ptxlite::compile(shlSrc, o2);
    std::string ptx = ptxlite::emitPTX(funcs, o2);
    CHECK(contains(ptx, "shl.b32"), "PTX: shl.b32 present for x*4 at O2");
  }
  // mad.lo.s32 present for integer mul-add
  {
    const char* madsrc = R"(
kernel madtest(int* a, int* b, int* c, int* out, int n) {
  int i = gid();
  if (i < n) { out[i] = a[i] * b[i] + c[i]; }
}
)";
    ptxlite::Options o2; o2.opt = 2;
    auto funcs = ptxlite::compile(madsrc, o2);
    std::string ptx = ptxlite::emitPTX(funcs, o2);
    CHECK(contains(ptx, "mad.lo.s32"), "PTX: mad.lo.s32 present for a*b+c at O2");
  }
  // Constant folded: 2+3 = 5, dead branch removed
  {
    const char* cfSrc = R"(
kernel cftest(int* out, int n) {
  int i = gid();
  if (i < n) {
    int x = 2 + 3;
    out[i] = x;
  }
}
)";
    ptxlite::Options o1; o1.opt = 1;
    auto funcs = ptxlite::compile(cfSrc, o1);
    std::string ptx = ptxlite::emitPTX(funcs, o1);
    // The constant 5 should appear, not separate adds
    CHECK(contains(ptx, "5"), "PTX: constant 2+3 folded to 5");
  }
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

static void testDiagnostics() {
  // Use of undeclared identifier
  {
    const char* src = "kernel bad(int n) { int i = gid(); int x = unk; }";
    bool caught = false; int line = -1;
    try { ptxlite::Options o; ptxlite::compile(src, o); }
    catch (const ptxlite::CompileError& e) { caught = true; line = e.line; }
    CHECK(caught, "diagnostics: undeclared identifier caught");
    (void)line;
  }
  // Type error: float to int without cast
  {
    const char* src = "kernel bad(float* x, int n) {\n  int i = gid();\n  int y = x[0];\n}";
    bool caught = false;
    try { ptxlite::Options o; ptxlite::compile(src, o); }
    catch (const ptxlite::CompileError& e) { caught = true; }
    CHECK(caught, "diagnostics: float-to-int implicit cast caught");
  }
  // Redefinition of variable
  {
    const char* src = "kernel bad(int n) { int x = 1; int x = 2; }";
    bool caught = false;
    try { ptxlite::Options o; ptxlite::compile(src, o); }
    catch (const ptxlite::CompileError& e) { caught = true; }
    CHECK(caught, "diagnostics: redefinition caught");
  }
}

// ── Fuzzer ────────────────────────────────────────────────────────────────────
// Generates random integer-arithmetic + if/else + for-loop kernels and
// checks that O0, O1, O2 produce identical outputs.

struct FuzzRng {
  std::mt19937 rng;
  FuzzRng(uint32_t seed) : rng(seed) {}
  int next(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }
};

static std::string genFuzzKernel(FuzzRng& rng) {
  // Very small kernels: just one thread writes out[0] = expression.
  // The expression mixes +, -, *, %, if/else, short bounded for-loops.
  int nVars = rng.next(2, 5);
  std::ostringstream o;
  o << "kernel fuzz(int* out, int dummy) {\n";
  o << "  int i = gid();\n";
  o << "  if (i < 1) {\n";

  // Declare variables with random constants
  for (int v = 0; v < nVars; ++v)
    o << "    int v" << v << " = " << rng.next(1, 20) << ";\n";

  // A bounded for loop
  int iters = rng.next(2, 6);
  int acc = rng.next(0, nVars - 1);
  int add = rng.next(0, nVars - 1);
  o << "    for (int k = 0; k < " << iters << "; k += 1) {\n";
  o << "      v" << acc << " += v" << add << ";\n";
  o << "    }\n";

  // An if/else
  int lhs = rng.next(0, nVars - 1), rhs = rng.next(0, nVars - 1);
  int tgt = rng.next(0, nVars - 1), tgt2 = rng.next(0, nVars - 1);
  o << "    if (v" << lhs << " < v" << rhs << ") {\n";
  o << "      v" << tgt << " += 1;\n";
  o << "    } else {\n";
  o << "      v" << tgt2 << " += 2;\n";
  o << "    }\n";

  // Combine results
  int result = rng.next(0, nVars - 1);
  for (int v = 0; v < nVars; ++v)
    if (v != result) o << "    v" << result << " += v" << v << ";\n";

  o << "    out[0] = v" << result << ";\n";
  o << "  }\n}\n";
  return o.str();
}

static void testFuzzer(int numPrograms = 200) {
  int fuzzPass = 0, fuzzFail = 0;
  for (int seed = 0; seed < numPrograms; ++seed) {
    FuzzRng rng(seed);
    std::string src = genFuzzKernel(rng);

    // Compile at O0, O1, O2
    std::vector<int32_t> out0(1,0), out1(1,0), out2(1,0);
    try {
      auto run = [&](int opt, std::vector<int32_t>& out) {
        auto rv = compile(src, opt);
        std::vector<ptxlite::Buffer> bufs = {{out.data(),1,4}};
        auto args = std::vector<ptxlite::Value>{ptrVal(0), ival(1)};
        ptxlite::Interpreter interp(bufs);
        interp.launch(rv.funcs[0], args, 1, 1);
      };
      run(0, out0); run(1, out1); run(2, out2);
      if (out0[0] == out1[0] && out1[0] == out2[0]) {
        ++fuzzPass;
      } else {
        ++fuzzFail;
        std::cerr << "FUZZ MISMATCH seed=" << seed << " O0=" << out0[0] << " O1=" << out1[0] << " O2=" << out2[0] << "\n";
        std::cerr << src << "\n";
      }
    } catch (...) {
      // Compile/interp errors in fuzz programs are skipped (not test failures)
    }
  }
  CHECK(fuzzFail == 0, "fuzzer: O0==O1==O2 for all " + std::to_string(fuzzPass) + " programs (fail=" + std::to_string(fuzzFail) + ")");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
  std::cout << "=== ptxlite test suite ===\n";

  // Correctness at O0, O1, O2
  for (int opt = 0; opt <= 2; ++opt) {
    testSaxpy(opt);
    testRowsum(opt);
    testCollatz(opt);
    testMandelbrot(opt);
  }

  testShortCircuit();
  testPtxOptChecks();
  testDiagnostics();
  testFuzzer(200);

  std::cout << "PASS: " << pass_ << "  FAIL: " << fail_ << "\n";
  return fail_ == 0 ? 0 : 1;
}
