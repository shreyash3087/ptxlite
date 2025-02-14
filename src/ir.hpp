// Mid-level IR: typed virtual registers, three-address instructions, basic blocks.
//
// Registers are mutable (not SSA), mirroring PTX's own virtual registers. Every unused
// operand slot is -1, which lets generic code walk uses without a per-opcode table.
#pragma once
#include <sstream>
#include <string>
#include <vector>

namespace ptxlite {

enum class VT { S32, F32, Pred, U64 };
enum class Op { Const, Mov, Param, Special, Bin, Cmp, Neg, Not, Cast, Addr, Load, Store, Sqrt, MulAdd, Br, CondBr, Ret };
enum class IBin { Add, Sub, Mul, Div, Rem, Min, Max, Shl };
enum class ICmp { Lt, Le, Gt, Ge, Eq, Ne };
enum class Spec { ThreadIdx, BlockIdx, BlockDim };
enum class CastK { I2F, F2I };

struct Inst {
  Op op = Op::Ret;
  VT ty = VT::S32;  // result type; for Cmp the operand type; for Store the value type
  int dst = -1, a = -1, b = -1, c = -1;
  int32_t ii = 0;   // Const (S32/Pred payload), Param index, Addr element size
  float ff = 0;     // Const (F32 payload)
  int sub = 0;      // IBin / ICmp / Spec / CastK
  int t1 = -1, t2 = -1;  // branch targets (block ids)

  bool operator==(const Inst& o) const {
    return op == o.op && ty == o.ty && dst == o.dst && a == o.a && b == o.b && c == o.c && ii == o.ii &&
           (ff == o.ff || (ff != ff && o.ff != o.ff)) && sub == o.sub && t1 == o.t1 && t2 == o.t2;
  }
  bool operator!=(const Inst& o) const { return !(*this == o); }
};

inline bool isTerm(Op o) { return o == Op::Br || o == Op::CondBr || o == Op::Ret; }
inline bool hasDst(const Inst& i) { return !isTerm(i.op) && i.op != Op::Store; }
// Instructions with no side effects: removable when their result is unused.
inline bool isPure(const Inst& i) { return hasDst(i); }

template <class F>
void forEachUse(const Inst& i, F f) {
  if (i.a >= 0) f(i.a);
  if (i.b >= 0) f(i.b);
  if (i.c >= 0) f(i.c);
}

struct ParamInfo {
  std::string name;
  VT vt;     // register type holding the value (U64 for pointers)
  bool ptr;
  VT elem;   // element type for pointers, same as vt for scalars
};

struct Block {
  int id = 0;
  std::vector<Inst> insts;
};

struct Func {
  std::string name;
  std::vector<ParamInfo> params;
  std::vector<VT> regTy;
  std::vector<Block> blocks;  // layout order; blocks[0] is the entry

  int newReg(VT t) {
    regTy.push_back(t);
    return static_cast<int>(regTy.size()) - 1;
  }
  int indexOf(int id) const {
    for (size_t i = 0; i < blocks.size(); ++i)
      if (blocks[i].id == id) return static_cast<int>(i);
    return -1;
  }
  size_t instCount() const {
    size_t n = 0;
    for (auto& b : blocks) n += b.insts.size();
    return n;
  }
};

inline const char* vtName(VT t) {
  switch (t) {
    case VT::S32: return "s32";
    case VT::F32: return "f32";
    case VT::Pred: return "pred";
    default: return "u64";
  }
}

inline std::string toString(const Func& f) {
  static const char* bin[] = {"add", "sub", "mul", "div", "rem", "min", "max", "shl"};
  static const char* cmp[] = {"lt", "le", "gt", "ge", "eq", "ne"};
  static const char* spec[] = {"threadIdx", "blockIdx", "blockDim"};
  std::ostringstream o;
  o << "func " << f.name << "(";
  for (size_t i = 0; i < f.params.size(); ++i) {
    if (i) o << ", ";
    o << f.params[i].name << ":" << vtName(f.params[i].vt) << (f.params[i].ptr ? "*" : "");
  }
  o << ")\n";
  auto R = [&](int r) { return "%" + std::to_string(r) + ":" + vtName(f.regTy[r]); };
  for (auto& b : f.blocks) {
    o << " bb" << b.id << ":\n";
    for (auto& i : b.insts) {
      o << "   ";
      switch (i.op) {
        case Op::Const:
          o << R(i.dst) << " = const " << (i.ty == VT::F32 ? std::to_string(i.ff) : std::to_string(i.ii));
          break;
        case Op::Mov: o << R(i.dst) << " = mov " << R(i.a); break;
        case Op::Param: o << R(i.dst) << " = param " << i.ii; break;
        case Op::Special: o << R(i.dst) << " = " << spec[i.sub]; break;
        case Op::Bin: o << R(i.dst) << " = " << bin[i.sub] << " " << R(i.a) << ", " << R(i.b); break;
        case Op::Cmp: o << R(i.dst) << " = cmp." << cmp[i.sub] << "." << vtName(i.ty) << " " << R(i.a) << ", " << R(i.b); break;
        case Op::Neg: o << R(i.dst) << " = neg " << R(i.a); break;
        case Op::Not: o << R(i.dst) << " = not " << R(i.a); break;
        case Op::Cast: o << R(i.dst) << " = " << (i.sub == 0 ? "i2f " : "f2i ") << R(i.a); break;
        case Op::Addr: o << R(i.dst) << " = addr " << R(i.a) << " + " << R(i.b) << "*" << i.ii; break;
        case Op::Load: o << R(i.dst) << " = load " << R(i.a); break;
        case Op::Store: o << "store " << R(i.a) << ", " << R(i.b); break;
        case Op::Sqrt: o << R(i.dst) << " = sqrt " << R(i.a); break;
        case Op::MulAdd: o << R(i.dst) << " = muladd " << R(i.a) << ", " << R(i.b) << ", " << R(i.c); break;
        case Op::Br: o << "br bb" << i.t1; break;
        case Op::CondBr: o << "condbr " << R(i.a) << ", bb" << i.t1 << ", bb" << i.t2; break;
        case Op::Ret: o << "ret"; break;
      }
      o << "\n";
    }
  }
  return o.str();
}

}  // namespace ptxlite
