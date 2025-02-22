// PTX emitter: translates each IR Func to PTX assembly.
//
// Conventions used:
//   - .version 7.0 / .target from Options::arch / .address_size 64
//   - Pointer params: ld.param.u64 then cvta.to.global.u64
//   - Integer registers declared as .b32 (%r<N>), float as .f32 (%f<N>),
//     predicates as .pred (%p<N>), 64-bit addresses as .b64 (%rd<N>).
//   - Addresses: mad.wide.s32 to compute byte address from base + index*4.
//   - Float constants: "0f" followed by 8 uppercase hex digits.
//   - Labels: $L__BB<kernel>_<block>.
//   - An unconditional branch is omitted when the target is the next block.
//   - CondBr whose true target falls through is emitted as @!%p bra.
#pragma once
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "ir.hpp"
#include "semantics.hpp"

namespace ptxlite {

class PtxEmitter {
 public:
  std::string emit(const std::vector<Func>& funcs, const Options& opts) {
    std::ostringstream o;
    o << ".version 7.0\n";
    o << ".target " << opts.arch << "\n";
    o << ".address_size 64\n\n";
    for (auto& f : funcs) emitFunc(o, f);
    return o.str();
  }

 private:
  // Per-function register maps: IR register index -> per-type dense index.
  std::vector<int> rIdx_;   // %r  (S32)
  std::vector<int> fIdx_;   // %f  (F32)
  std::vector<int> pIdx_;   // %p  (Pred)
  std::vector<int> rdIdx_;  // %rd (U64)
  int nR_ = 0, nF_ = 0, nP_ = 0, nRd_ = 0;
  std::string kname_;

  void assignRegs(const Func& f) {
    size_t n = f.regTy.size();
    rIdx_.assign(n, -1);
    fIdx_.assign(n, -1);
    pIdx_.assign(n, -1);
    rdIdx_.assign(n, -1);
    nR_ = nF_ = nP_ = nRd_ = 0;
    for (size_t i = 0; i < n; ++i) {
      switch (f.regTy[i]) {
        case VT::S32: rIdx_[i] = nR_++; break;
        case VT::F32: fIdx_[i] = nF_++; break;
        case VT::Pred: pIdx_[i] = nP_++; break;
        case VT::U64: rdIdx_[i] = nRd_++; break;
      }
    }
  }

  std::string reg(int r) const {
    if (rIdx_[r] >= 0) return "%r" + std::to_string(rIdx_[r]);
    if (fIdx_[r] >= 0) return "%f" + std::to_string(fIdx_[r]);
    if (pIdx_[r] >= 0) return "%p" + std::to_string(pIdx_[r]);
    return "%rd" + std::to_string(rdIdx_[r]);
  }

  std::string label(int blockId) const {
    return "$L__BB" + kname_ + "_" + std::to_string(blockId);
  }

  static std::string fconst(float f) {
    uint32_t u = floatBits(f);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0f%08X", u);
    return buf;
  }

  void emitFunc(std::ostringstream& o, const Func& f) {
    kname_ = f.name;
    assignRegs(f);

    // .visible .entry declaration
    o << ".visible .entry " << f.name << "(\n";
    for (size_t i = 0; i < f.params.size(); ++i) {
      if (i) o << ",\n";
      o << "\t.param .u64 " << f.name << "_param_" << i;
    }
    o << "\n)\n{\n";

    // Register declarations
    if (nR_ > 0) o << "\t.reg .b32 \t%r<" << nR_ << ">;\n";
    if (nF_ > 0) o << "\t.reg .f32 \t%f<" << nF_ << ">;\n";
    if (nP_ > 0) o << "\t.reg .pred \t%p<" << nP_ << ">;\n";
    if (nRd_ > 0) o << "\t.reg .b64 \t%rd<" << nRd_ << ">;\n";
    o << "\n";

    // Emit each block
    for (size_t bi = 0; bi < f.blocks.size(); ++bi) {
      const Block& b = f.blocks[bi];
      // Label (skip the entry block label – PTX does not need it)
      if (bi > 0) o << label(b.id) << ":\n";

      for (size_t ii = 0; ii < b.insts.size(); ++ii) {
        const Inst& in = b.insts[ii];
        // Determine the next block id for fall-through suppression
        int nextBid = -1;
        if (ii + 1 == b.insts.size() && bi + 1 < f.blocks.size())
          nextBid = f.blocks[bi + 1].id;

        emitInst(o, f, in, nextBid);
      }
    }
    o << "}\n\n";
  }

  void emitInst(std::ostringstream& o, const Func& f, const Inst& in, int nextBid) {
    auto ind = [&]() -> std::ostringstream& { o << "\t"; return o; };
    switch (in.op) {
      case Op::Param: {
        // Load parameter (always u64 for the raw address)
        const ParamInfo& pi = f.params[in.ii];
        std::string pname = f.name + "_param_" + std::to_string(in.ii);
        if (pi.ptr) {
          // Pointer: load u64 then cvta.to.global
          ind() << "ld.param.u64 \t" << reg(in.dst) << ", [" << pname << "];\n";
          ind() << "cvta.to.global.u64 \t" << reg(in.dst) << ", " << reg(in.dst) << ";\n";
        } else if (pi.vt == VT::S32) {
          ind() << "ld.param.u32 \t" << reg(in.dst) << ", [" << pname << "];\n";
        } else {
          // F32: PTX uses .f32 param
          ind() << "ld.param.f32 \t" << reg(in.dst) << ", [" << pname << "];\n";
        }
        break;
      }
      case Op::Const: {
        if (in.ty == VT::F32) {
          ind() << "mov.f32 \t" << reg(in.dst) << ", " << fconst(in.ff) << ";\n";
        } else if (in.ty == VT::Pred) {
          // constant predicate
          if (in.ii) ind() << "setp.eq.s32 \t" << reg(in.dst) << ", 1, 1;\n";
          else ind() << "setp.ne.s32 \t" << reg(in.dst) << ", 1, 1;\n";
        } else {
          ind() << "mov.s32 \t" << reg(in.dst) << ", " << in.ii << ";\n";
        }
        break;
      }
      case Op::Mov: {
        if (in.ty == VT::F32)
          ind() << "mov.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ";\n";
        else if (in.ty == VT::Pred)
          ind() << "mov.pred \t" << reg(in.dst) << ", " << reg(in.a) << ";\n";
        else if (in.ty == VT::U64)
          ind() << "mov.b64 \t" << reg(in.dst) << ", " << reg(in.a) << ";\n";
        else
          ind() << "mov.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ";\n";
        break;
      }
      case Op::Special: {
        static const char* ntid[] = {"ntid.x", "ctaid.x", "ntid.x"};
        // threadIdx -> %tid.x, blockIdx -> %ctaid.x, blockDim -> %ntid.x
        const char* special_names[] = {"%tid.x", "%ctaid.x", "%ntid.x"};
        ind() << "mov.u32 \t" << reg(in.dst) << ", " << special_names[in.sub] << ";\n";
        (void)ntid;
        break;
      }
      case Op::Bin: {
        IBin op = static_cast<IBin>(in.sub);
        emitBin(o, in, op);
        break;
      }
      case Op::Cmp: {
        ICmp cmp = static_cast<ICmp>(in.sub);
        static const char* icmpOp[] = {"lt", "le", "gt", "ge", "eq", "ne"};
        // float != -> neu
        if (in.ty == VT::F32 && cmp == ICmp::Ne) {
          ind() << "setp.neu.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n";
        } else if (in.ty == VT::F32) {
          ind() << "setp." << icmpOp[static_cast<int>(cmp)] << ".f32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n";
        } else {
          ind() << "setp." << icmpOp[static_cast<int>(cmp)] << ".s32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n";
        }
        break;
      }
      case Op::Neg: {
        if (in.ty == VT::F32)
          ind() << "neg.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ";\n";
        else
          ind() << "neg.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ";\n";
        break;
      }
      case Op::Not: {
        ind() << "not.pred \t" << reg(in.dst) << ", " << reg(in.a) << ";\n";
        break;
      }
      case Op::Cast: {
        if (in.sub == static_cast<int>(CastK::I2F))
          ind() << "cvt.rn.f32.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ";\n";
        else
          ind() << "cvt.rzi.s32.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ";\n";
        break;
      }
      case Op::Addr: {
        // mad.wide.s32 rd, idx.s32, elem_size, base.u64
        ind() << "mad.wide.s32 \t" << reg(in.dst) << ", " << reg(in.b) << ", " << in.ii << ", " << reg(in.a) << ";\n";
        break;
      }
      case Op::Load: {
        if (in.ty == VT::F32)
          ind() << "ld.global.f32 \t" << reg(in.dst) << ", [" << reg(in.a) << "];\n";
        else
          ind() << "ld.global.s32 \t" << reg(in.dst) << ", [" << reg(in.a) << "];\n";
        break;
      }
      case Op::Store: {
        if (in.ty == VT::F32)
          ind() << "st.global.f32 \t[" << reg(in.a) << "], " << reg(in.b) << ";\n";
        else
          ind() << "st.global.s32 \t[" << reg(in.a) << "], " << reg(in.b) << ";\n";
        break;
      }
      case Op::Sqrt: {
        ind() << "sqrt.rn.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ";\n";
        break;
      }
      case Op::MulAdd: {
        if (in.ty == VT::F32)
          ind() << "fma.rn.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ", " << reg(in.c) << ";\n";
        else
          ind() << "mad.lo.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ", " << reg(in.c) << ";\n";
        break;
      }
      case Op::Br: {
        if (in.t1 != nextBid)
          ind() << "bra.uni \t" << label(in.t1) << ";\n";
        break;
      }
      case Op::CondBr: {
        // If true-target falls through, invert the predicate
        if (in.t1 == nextBid) {
          ind() << "@!" << reg(in.a) << " bra \t" << label(in.t2) << ";\n";
        } else if (in.t2 == nextBid) {
          ind() << "@" << reg(in.a) << " bra \t" << label(in.t1) << ";\n";
        } else {
          ind() << "@" << reg(in.a) << " bra \t" << label(in.t1) << ";\n";
          ind() << "bra.uni \t" << label(in.t2) << ";\n";
        }
        break;
      }
      case Op::Ret: {
        ind() << "ret;\n";
        break;
      }
    }
  }

  void emitBin(std::ostringstream& o, const Inst& in, IBin op) {
    auto ind = [&]() -> std::ostringstream& { o << "\t"; return o; };
    if (in.ty == VT::F32) {
      switch (op) {
        case IBin::Add: ind() << "add.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Sub: ind() << "sub.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Mul: ind() << "mul.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Div: ind() << "div.rn.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Min: ind() << "min.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Max: ind() << "max.f32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        default: break;
      }
    } else {  // S32
      switch (op) {
        case IBin::Add: ind() << "add.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Sub: ind() << "sub.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Mul: ind() << "mul.lo.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Div: ind() << "div.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Rem: ind() << "rem.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Min: ind() << "min.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Max: ind() << "max.s32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
        case IBin::Shl: ind() << "shl.b32 \t" << reg(in.dst) << ", " << reg(in.a) << ", " << reg(in.b) << ";\n"; break;
      }
    }
  }
};

}  // namespace ptxlite
