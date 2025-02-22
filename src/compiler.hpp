// Compiler driver: lex -> parse -> sema -> lower -> optimize -> emit.
//
// compile() accepts a source string and returns the optimized Func vector.
// emitPTX() runs the PTX emitter on the result.
// Both functions are separated so tests can inspect the IR before emission.
#pragma once
#include <string>
#include <vector>

#include "common.hpp"
#include "ir.hpp"
#include "lexer.hpp"
#include "lower.hpp"
#include "opt.hpp"
#include "parser.hpp"
#include "ptx.hpp"
#include "sema.hpp"

namespace ptxlite {

// compile: lex, parse, check, lower, optimize.
// If stats is non-null the per-kernel Stats are accumulated into *stats.
inline std::vector<Func> compile(const std::string& src, const Options& opts,
                                  Stats* stats = nullptr) {
  auto toks = Lexer(src).run();
  auto prog = Parser(std::move(toks)).parseProgram();
  Sema().checkProgram(prog);

  std::vector<Func> funcs;
  funcs.reserve(prog.kernels.size());
  for (const Kernel& k : prog.kernels) {
    Func f = Lowerer().lower(k);
    Stats st;
    st.instsBefore = f.instCount();
    optimize(f, opts, st);
    st.instsAfter = f.instCount();
    if (stats) stats->add(st);
    funcs.push_back(std::move(f));
  }
  return funcs;
}

// emitPTX: run the PTX emitter over already-compiled funcs.
inline std::string emitPTX(const std::vector<Func>& funcs, const Options& opts) {
  return PtxEmitter().emit(funcs, opts);
}

}  // namespace ptxlite
