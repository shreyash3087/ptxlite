// Shared types: diagnostics, the source-level type system, and compiler options.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ptxlite {

struct CompileError : std::runtime_error {
  int line, col;
  CompileError(const std::string& msg, int l, int c) : std::runtime_error(msg), line(l), col(c) {}
};

enum class Base { Int, Float, Bool, Void };

struct Type {
  Base base = Base::Void;
  bool ptr = false;
  bool operator==(const Type& o) const { return base == o.base && ptr == o.ptr; }
  bool operator!=(const Type& o) const { return !(*this == o); }
  bool numeric() const { return !ptr && (base == Base::Int || base == Base::Float); }
  std::string str() const {
    std::string s = base == Base::Int ? "int" : base == Base::Float ? "float" : base == Base::Bool ? "bool" : "void";
    return ptr ? s + "*" : s;
  }
};
inline Type tInt() { return {Base::Int, false}; }
inline Type tFloat() { return {Base::Float, false}; }
inline Type tBool() { return {Base::Bool, false}; }

struct Options {
  int opt = 2;                 // 0 = no optimization, 1 = scalar opts, 2 = + mul-add fusion
  bool fma = true;             // allow contraction of float a*b+c into fma.rn.f32 at -O2
  std::string arch = "sm_75";  // PTX .target
};

}  // namespace ptxlite
