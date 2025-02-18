// Optimization passes over the IR.
//
//   simplifyLocal  constant folding, copy propagation, algebraic identities, strength
//                  reduction (x * 2^k -> x << k), branch folding. Single-definition constant
//                  registers are propagated across blocks.
//   cleanupCFG     unreachable-block removal, empty-block forwarding, straight-line merging.
//   dce            dead-code elimination driven by global use counts, iterated to a fixpoint.
//   fuseMulAdd     peephole fusion of mul+add into mad.lo.s32 / fma.rn.f32 (-O2).
#pragma once
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "common.hpp"
#include "ir.hpp"
#include "semantics.hpp"

namespace ptxlite {

struct Stats {
  int folded = 0;       // instructions replaced by constants
  int simplified = 0;   // algebraic identities applied
  int strength = 0;     // multiplications turned into shifts
  int copies = 0;       // operands rewritten by copy propagation
  int branches = 0;     // conditional branches folded
  int blocksRemoved = 0;
  int merged = 0;       // straight-line block merges
  int dead = 0;         // instructions removed by DCE
  int fused = 0;        // mul+add pairs fused
  size_t instsBefore = 0, instsAfter = 0;

  void add(const Stats& o) {
    folded += o.folded; simplified += o.simplified; strength += o.strength; copies += o.copies;
    branches += o.branches; blocksRemoved += o.blocksRemoved; merged += o.merged; dead += o.dead;
    fused += o.fused; instsBefore += o.instsBefore; instsAfter += o.instsAfter;
  }
};

namespace detail {

inline Inst makeConstI(int dst, VT ty, int32_t v) {
  Inst c;
  c.op = Op::Const; c.ty = ty; c.dst = dst; c.ii = v;
  return c;
}
inline Inst makeConstF(int dst, float v) {
  Inst c;
  c.op = Op::Const; c.ty = VT::F32; c.dst = dst; c.ff = v;
  return c;
}
inline Inst makeMov(int dst, VT ty, int src) {
  Inst m;
  m.op = Op::Mov; m.ty = ty; m.dst = dst; m.a = src;
  return m;
}
inline bool isPow2(int32_t v) { return v > 1 && (v & (v - 1)) == 0; }
inline int log2i(int32_t v) {
  int k = 0;
  while ((1 << k) < v) ++k;
  return k;
}
inline bool commutative(IBin op) { return op == IBin::Add || op == IBin::Mul || op == IBin::Min || op == IBin::Max; }

}  // namespace detail

inline bool simplifyLocal(Func& f, Stats& st) {
  using namespace detail;
  bool changed = false;
  const size_t nOrig = f.regTy.size();

  // Registers with exactly one definition that is a Const hold that value everywhere they are read.
  std::vector<int> defCnt(nOrig, 0);
  std::vector<Inst> lastDef(nOrig);
  for (auto& b : f.blocks)
    for (auto& i : b.insts)
      if (hasDst(i)) { ++defCnt[i.dst]; lastDef[i.dst] = i; }

  for (auto& blk : f.blocks) {
    std::unordered_map<int, Inst> kc;    // reg -> Const inst known inside this block
    std::unordered_map<int, int> cp;     // reg -> register it is a copy of
    auto getConst = [&](int r, Inst& out) -> bool {
      if (r < 0) return false;
      auto it = kc.find(r);
      if (it != kc.end()) { out = it->second; return true; }
      if (static_cast<size_t>(r) < nOrig && defCnt[r] == 1 && lastDef[r].op == Op::Const) { out = lastDef[r]; return true; }
      return false;
    };
    auto kill = [&](int d) {
      kc.erase(d);
      cp.erase(d);
      for (auto it = cp.begin(); it != cp.end();) {
        if (it->second == d) it = cp.erase(it);
        else ++it;
      }
    };

    std::vector<Inst> out;
    out.reserve(blk.insts.size());
    for (Inst in : blk.insts) {
      const Inst orig = in;

      // 1. copy propagation on operands
      for (int* p : {&in.a, &in.b, &in.c}) {
        if (*p < 0) continue;
        auto it = cp.find(*p);
        if (it != cp.end()) { *p = it->second; ++st.copies; }
      }

      Inst ka, kb, kcn;
      bool ca = getConst(in.a, ka), cb = getConst(in.b, kb), cc = getConst(in.c, kcn);

      switch (in.op) {
        case Op::Mov:
          if (ca) { in = ka; in.dst = orig.dst; ++st.folded; }
          break;

        case Op::Bin: {
          IBin op = static_cast<IBin>(in.sub);
          if (commutative(op) && ca && !cb) {  // keep constants on the right
            std::swap(in.a, in.b);
            std::swap(ka, kb);
            std::swap(ca, cb);
          }
          if (ca && cb) {
            if (in.ty == VT::S32) {
              if (canFoldI(op, kb.ii)) { in = makeConstI(in.dst, VT::S32, evalI(op, ka.ii, kb.ii)); ++st.folded; }
            } else {
              in = makeConstF(in.dst, evalF(op, ka.ff, kb.ff));
              ++st.folded;
            }
            break;
          }
          if (in.ty == VT::S32) {
            if (cb) {
              int32_t v = kb.ii;
              if ((op == IBin::Add || op == IBin::Sub || op == IBin::Shl) && v == 0) { in = makeMov(in.dst, VT::S32, in.a); ++st.simplified; }
              else if (op == IBin::Mul && v == 1) { in = makeMov(in.dst, VT::S32, in.a); ++st.simplified; }
              else if (op == IBin::Div && v == 1) { in = makeMov(in.dst, VT::S32, in.a); ++st.simplified; }
              else if (op == IBin::Mul && v == 0) { in = makeConstI(in.dst, VT::S32, 0); ++st.simplified; }
              else if (op == IBin::Mul && isPow2(v)) {
                int sh = f.newReg(VT::S32);
                Inst k = makeConstI(sh, VT::S32, log2i(v));
                out.push_back(k);
                kc[sh] = k;
                in.sub = static_cast<int>(IBin::Shl);
                in.b = sh;
                ++st.strength;
              }
            } else if (op == IBin::Sub && in.a == in.b) {
              in = makeConstI(in.dst, VT::S32, 0);
              ++st.simplified;
            }
          } else if (cb) {  // float identities that are exact for every input, including NaN and -0
            float v = kb.ff;
            bool posZero = v == 0.0f && !std::signbit(v);
            if ((op == IBin::Mul && v == 1.0f) || (op == IBin::Div && v == 1.0f) || (op == IBin::Sub && posZero)) {
              in = makeMov(in.dst, VT::F32, in.a);
              ++st.simplified;
            }
          }
          break;
        }

        case Op::Cmp:
          if (ca && cb) {
            bool r = in.ty == VT::S32 ? evalCmpI(static_cast<ICmp>(in.sub), ka.ii, kb.ii)
                                      : evalCmpF(static_cast<ICmp>(in.sub), ka.ff, kb.ff);
            in = makeConstI(in.dst, VT::Pred, r ? 1 : 0);
            ++st.folded;
          }
          break;

        case Op::Neg:
          if (ca) {
            in = in.ty == VT::S32 ? makeConstI(in.dst, VT::S32, evalI(IBin::Sub, 0, ka.ii)) : makeConstF(in.dst, -ka.ff);
            ++st.folded;
          }
          break;
        case Op::Not:
          if (ca) { in = makeConstI(in.dst, VT::Pred, ka.ii ? 0 : 1); ++st.folded; }
          break;
        case Op::Cast:
          if (ca) {
            in = in.sub == static_cast<int>(CastK::I2F) ? makeConstF(in.dst, static_cast<float>(ka.ii))
                                                         : makeConstI(in.dst, VT::S32, f2i(ka.ff));
            ++st.folded;
          }
          break;
        case Op::Sqrt:
          if (ca) { in = makeConstF(in.dst, std::sqrt(ka.ff)); ++st.folded; }
          break;
        case Op::MulAdd:
          if (ca && cb && cc) {
            in = in.ty == VT::S32 ? makeConstI(in.dst, VT::S32, evalMulAddI(ka.ii, kb.ii, kcn.ii))
                                  : makeConstF(in.dst, evalMulAddF(ka.ff, kb.ff, kcn.ff));
            ++st.folded;
          }
          break;

        case Op::CondBr:
          if (ca) {
            int target = ka.ii ? in.t1 : in.t2;
            in = Inst();
            in.op = Op::Br;
            in.t1 = target;
            ++st.branches;
          } else if (in.t1 == in.t2) {
            int target = in.t1;
            in = Inst();
            in.op = Op::Br;
            in.t1 = target;
            ++st.branches;
          }
          break;
        default: break;
      }

      // 2. maintain known-constant and copy facts
      if (hasDst(in)) {
        kill(in.dst);
        if (in.op == Op::Const) kc[in.dst] = in;
        else if (in.op == Op::Mov && in.a != in.dst) cp[in.dst] = in.a;
      }
      if (in != orig) changed = true;
      out.push_back(in);
    }
    blk.insts = std::move(out);
  }
  return changed;
}

inline bool removeUnreachable(Func& f, Stats& st) {
  std::unordered_set<int> seen;
  std::vector<int> work{f.blocks[0].id};
  while (!work.empty()) {
    int id = work.back();
    work.pop_back();
    if (!seen.insert(id).second) continue;
    const Block& b = f.blocks[f.indexOf(id)];
    const Inst& t = b.insts.back();
    if (t.op == Op::Br) work.push_back(t.t1);
    else if (t.op == Op::CondBr) { work.push_back(t.t1); work.push_back(t.t2); }
  }
  size_t before = f.blocks.size();
  f.blocks.erase(std::remove_if(f.blocks.begin(), f.blocks.end(), [&](const Block& b) { return !seen.count(b.id); }), f.blocks.end());
  st.blocksRemoved += static_cast<int>(before - f.blocks.size());
  return before != f.blocks.size();
}

inline bool cleanupCFG(Func& f, Stats& st) {
  bool changed = removeUnreachable(f, st);

  // Forward jumps through blocks that contain nothing but an unconditional branch.
  std::unordered_map<int, int> fwd;
  for (size_t i = 1; i < f.blocks.size(); ++i) {
    const Block& b = f.blocks[i];
    if (b.insts.size() == 1 && b.insts[0].op == Op::Br && b.insts[0].t1 != b.id) fwd[b.id] = b.insts[0].t1;
  }
  auto resolve = [&](int id) {
    for (int hops = 0; hops < 64; ++hops) {
      auto it = fwd.find(id);
      if (it == fwd.end()) break;
      id = it->second;
    }
    return id;
  };
  for (auto& b : f.blocks) {
    Inst& t = b.insts.back();
    if (t.op == Op::Br || t.op == Op::CondBr) {
      int n1 = resolve(t.t1);
      int n2 = t.op == Op::CondBr ? resolve(t.t2) : t.t2;
      if (n1 != t.t1 || n2 != t.t2) { t.t1 = n1; t.t2 = n2; changed = true; }
    }
  }
  changed |= removeUnreachable(f, st);

  // Merge a block into its unique predecessor when that predecessor ends in an unconditional branch.
  for (bool again = true; again;) {
    again = false;
    std::unordered_map<int, int> preds;
    for (auto& b : f.blocks) {
      const Inst& t = b.insts.back();
      if (t.op == Op::Br) ++preds[t.t1];
      else if (t.op == Op::CondBr) { ++preds[t.t1]; ++preds[t.t2]; }
    }
    for (size_t i = 0; i < f.blocks.size(); ++i) {
      const Inst last = f.blocks[i].insts.back();
      if (last.op != Op::Br || last.t1 == f.blocks[i].id || preds[last.t1] != 1) continue;
      int j = f.indexOf(last.t1);
      if (j <= 0) continue;
      f.blocks[i].insts.pop_back();
      auto tail = std::move(f.blocks[j].insts);
      f.blocks[i].insts.insert(f.blocks[i].insts.end(), tail.begin(), tail.end());
      f.blocks.erase(f.blocks.begin() + j);
      ++st.merged;
      changed = again = true;
      break;
    }
  }
  return changed;
}

inline bool dce(Func& f, Stats& st) {
  bool any = false;
  for (;;) {
    std::vector<int> uses(f.regTy.size(), 0);
    for (auto& b : f.blocks)
      for (auto& i : b.insts) forEachUse(i, [&](int r) { ++uses[r]; });
    bool changed = false;
    for (auto& b : f.blocks) {
      size_t before = b.insts.size();
      b.insts.erase(std::remove_if(b.insts.begin(), b.insts.end(),
                                   [&](const Inst& i) {
                                     if (i.op == Op::Mov && i.a == i.dst) return true;
                                     return isPure(i) && uses[i.dst] == 0;
                                   }),
                    b.insts.end());
      if (b.insts.size() != before) {
        st.dead += static_cast<int>(before - b.insts.size());
        changed = true;
      }
    }
    if (!changed) break;
    any = true;
  }
  return any;
}

// t = a*b ; d = t + c   ==>   d = a*b + c, when t has no other use and a, b are not redefined in between.
inline bool fuseMulAdd(Func& f, bool fuseFloat, Stats& st) {
  std::vector<int> uses(f.regTy.size(), 0), defs(f.regTy.size(), 0);
  for (auto& b : f.blocks)
    for (auto& i : b.insts) {
      forEachUse(i, [&](int r) { ++uses[r]; });
      if (hasDst(i)) ++defs[i.dst];
    }
  bool changed = false;
  for (auto& blk : f.blocks) {
    auto& v = blk.insts;
    for (size_t j = 0; j < v.size(); ++j) {
      Inst add = v[j];
      if (add.op != Op::Bin || add.sub != static_cast<int>(IBin::Add)) continue;
      if (add.ty == VT::F32 && !fuseFloat) continue;
      for (int side = 0; side < 2; ++side) {
        int t = side == 0 ? add.a : add.b;
        int other = side == 0 ? add.b : add.a;
        if (t == other || defs[t] != 1 || uses[t] != 1) continue;
        int i = -1;
        for (int k = static_cast<int>(j) - 1; k >= 0; --k)
          if (hasDst(v[k]) && v[k].dst == t) { i = k; break; }
        if (i < 0) continue;
        const Inst mul = v[i];
        if (mul.op != Op::Bin || mul.sub != static_cast<int>(IBin::Mul) || mul.ty != add.ty) continue;
        bool clobbered = false;
        for (size_t k = i + 1; k < j && !clobbered; ++k)
          if (hasDst(v[k]) && (v[k].dst == mul.a || v[k].dst == mul.b)) clobbered = true;
        if (clobbered) continue;
        Inst fz;
        fz.op = Op::MulAdd; fz.ty = add.ty; fz.dst = add.dst; fz.a = mul.a; fz.b = mul.b; fz.c = other;
        v[j] = fz;
        v.erase(v.begin() + i);
        --j;
        ++st.fused;
        changed = true;
        break;
      }
    }
  }
  return changed;
}

inline void optimize(Func& f, const Options& o, Stats& st) {
  if (o.opt <= 0) return;
  for (int round = 0; round < 12; ++round) {
    bool ch = false;
    ch |= simplifyLocal(f, st);
    ch |= cleanupCFG(f, st);
    ch |= dce(f, st);
    if (!ch) break;
  }
  if (o.opt >= 2 && fuseMulAdd(f, o.fma, st)) dce(f, st);
}

}  // namespace ptxlite
