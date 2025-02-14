// Arithmetic semantics shared by the constant folder and the IR interpreter.
//
// Keeping one definition guarantees that folding at compile time and executing at run
// time can never disagree, so differential tests only exercise the transformations.
#pragma once
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "ir.hpp"

namespace ptxlite {

// 32-bit integer ops wrap on overflow. Division by zero yields 0 (PTX leaves it undefined);
// INT_MIN / -1 is defined as INT_MIN, INT_MIN % -1 as 0.
inline int32_t evalI(IBin op, int32_t a, int32_t b) {
  uint32_t ua = static_cast<uint32_t>(a), ub = static_cast<uint32_t>(b);
  switch (op) {
    case IBin::Add: return static_cast<int32_t>(ua + ub);
    case IBin::Sub: return static_cast<int32_t>(ua - ub);
    case IBin::Mul: return static_cast<int32_t>(ua * ub);
    case IBin::Div:
      if (b == 0) return 0;
      if (a == INT_MIN && b == -1) return INT_MIN;
      return a / b;
    case IBin::Rem:
      if (b == 0) return 0;
      if (a == INT_MIN && b == -1) return 0;
      return a % b;
    case IBin::Min: return a < b ? a : b;
    case IBin::Max: return a > b ? a : b;
    case IBin::Shl: return (b < 0 || b >= 32) ? 0 : static_cast<int32_t>(ua << b);
  }
  return 0;
}

// Folding a division by zero would bake in an arbitrary value, so the folder skips it.
inline bool canFoldI(IBin op, int32_t b) { return !((op == IBin::Div || op == IBin::Rem) && b == 0); }

inline float evalF(IBin op, float a, float b) {
  switch (op) {
    case IBin::Add: return a + b;
    case IBin::Sub: return a - b;
    case IBin::Mul: return a * b;
    case IBin::Div: return a / b;
    case IBin::Min: return std::fmin(a, b);
    case IBin::Max: return std::fmax(a, b);
    default: return 0.0f;
  }
}

inline bool evalCmpI(ICmp c, int32_t a, int32_t b) {
  switch (c) {
    case ICmp::Lt: return a < b;
    case ICmp::Le: return a <= b;
    case ICmp::Gt: return a > b;
    case ICmp::Ge: return a >= b;
    case ICmp::Eq: return a == b;
    case ICmp::Ne: return a != b;
  }
  return false;
}

// Ordered comparisons are false when either side is NaN; != is true (PTX setp.neu).
inline bool evalCmpF(ICmp c, float a, float b) {
  switch (c) {
    case ICmp::Lt: return a < b;
    case ICmp::Le: return a <= b;
    case ICmp::Gt: return a > b;
    case ICmp::Ge: return a >= b;
    case ICmp::Eq: return a == b;
    case ICmp::Ne: return a != b;
  }
  return false;
}

// float -> int truncates toward zero, saturates, and maps NaN to 0 (PTX cvt.rzi.s32.f32).
inline int32_t f2i(float f) {
  if (f != f) return 0;
  if (f >= 2147483648.0f) return INT_MAX;
  if (f <= -2147483648.0f) return INT_MIN;
  return static_cast<int32_t>(f);
}

inline int32_t evalMulAddI(int32_t a, int32_t b, int32_t c) { return evalI(IBin::Add, evalI(IBin::Mul, a, b), c); }
inline float evalMulAddF(float a, float b, float c) { return std::fmaf(a, b, c); }

inline uint32_t floatBits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
inline float bitsFloat(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace ptxlite
