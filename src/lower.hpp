// AST -> IR lowering. Structured control flow becomes basic blocks joined by Br/CondBr;
// && and || become short-circuit control flow so guards like `i < n && x[i] > 0` are safe.
#pragma once
#include <map>

#include "ast.hpp"
#include "ir.hpp"

namespace ptxlite {

class Lowerer {
 public:
  Func lower(const Kernel& k) {
    f_ = Func();
    f_.name = k.name;
    isVar_.clear();
    scopes_.clear();
    scopes_.emplace_back();
    nextBlock_ = 0;
    place(create());  // entry block

    for (size_t i = 0; i < k.params.size(); ++i) {
      const Param& p = k.params[i];
      ParamInfo pi;
      pi.name = p.name;
      pi.ptr = p.ty.ptr;
      pi.elem = vtOf(Type{p.ty.base, false});
      pi.vt = pi.ptr ? VT::U64 : pi.elem;
      f_.params.push_back(pi);
      int r = newReg(pi.vt, true);
      Inst in = mk(Op::Param, pi.vt);
      in.dst = r;
      in.ii = static_cast<int32_t>(i);
      emit(in);
      scopes_.back()[p.name] = r;
    }

    lowerBlock(k.body);
    if (!terminated()) emit(mk(Op::Ret, VT::S32));
    return std::move(f_);
  }

 private:
  Func f_;
  std::vector<bool> isVar_;  // true for registers that back a source variable (or a merged result)
  std::vector<std::map<std::string, int>> scopes_;
  int nextBlock_ = 0;

  static VT vtOf(Type t) {
    switch (t.base) {
      case Base::Int: return VT::S32;
      case Base::Float: return VT::F32;
      default: return VT::Pred;
    }
  }
  static Inst mk(Op op, VT ty) {
    Inst i;
    i.op = op;
    i.ty = ty;
    return i;
  }

  int newReg(VT t, bool isVar) {
    int r = f_.newReg(t);
    isVar_.resize(f_.regTy.size(), false);
    isVar_[r] = isVar;
    return r;
  }
  int create() { return nextBlock_++; }
  void place(int id) {
    Block b;
    b.id = id;
    f_.blocks.push_back(std::move(b));
  }
  void emit(const Inst& i) { f_.blocks.back().insts.push_back(i); }
  bool terminated() const {
    const auto& v = f_.blocks.back().insts;
    return !v.empty() && isTerm(v.back().op);
  }
  void emitBr(int target) {
    Inst i = mk(Op::Br, VT::S32);
    i.t1 = target;
    emit(i);
  }
  void emitCondBr(int pred, int t, int f) {
    Inst i = mk(Op::CondBr, VT::Pred);
    i.a = pred; i.t1 = t; i.t2 = f;
    emit(i);
  }

  int lookup(const std::string& n) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      auto f = it->find(n);
      if (f != it->end()) return f->second;
    }
    return -1;  // unreachable after Sema
  }

  // dst = src. If src is a fresh temporary produced by the previous instruction, retarget that
  // instruction to write dst directly instead of emitting a copy.
  void assign(int dst, int src) {
    auto& v = f_.blocks.back().insts;
    if (dst == src) return;
    if (isVar_[dst] && !isVar_[src] && !v.empty() && !isTerm(v.back().op) && hasDst(v.back()) && v.back().dst == src) {
      v.back().dst = dst;
      return;
    }
    Inst m = mk(Op::Mov, f_.regTy[dst]);
    m.dst = dst; m.a = src;
    emit(m);
  }

  void lowerBlock(const std::vector<StmtP>& b) {
    scopes_.emplace_back();
    for (auto& s : b) lowerStmt(*s);
    scopes_.pop_back();
  }

  void lowerStmt(const Stmt& s) {
    if (terminated()) place(create());  // code after a return/branch lands in an unreachable block
    switch (s.kind) {
      case SK::Decl: {
        int r;
        if (s.e1) {
          int v = lowerExpr(*s.e1);
          r = newReg(vtOf(s.declTy), true);
          assign(r, v);
        } else {
          r = newReg(vtOf(s.declTy), true);
          Inst c = mk(Op::Const, vtOf(s.declTy));
          c.dst = r;
          emit(c);  // zero-initialized
        }
        scopes_.back()[s.name] = r;
        break;
      }
      case SK::Assign: {
        int v = lowerExpr(*s.e1);
        assign(lookup(s.name), v);
        break;
      }
      case SK::Store: {
        int base = lookup(s.name);
        int idx = lowerExpr(*s.e1);
        int val = lowerExpr(*s.e2);
        int addr = emitAddr(base, idx);
        Inst st = mk(Op::Store, f_.regTy[val]);
        st.a = addr; st.b = val;
        emit(st);
        break;
      }
      case SK::If: {
        int c = lowerExpr(*s.e1);
        int thenB = create(), joinB = create();
        int elseB = s.other.empty() ? joinB : create();
        emitCondBr(c, thenB, elseB);
        place(thenB);
        lowerBlock(s.body);
        if (!terminated()) emitBr(joinB);
        if (!s.other.empty()) {
          place(elseB);
          lowerBlock(s.other);
          if (!terminated()) emitBr(joinB);
        }
        place(joinB);
        break;
      }
      case SK::While: {
        int hdr = create(), body = create(), exit = create();
        emitBr(hdr);
        place(hdr);
        int c = lowerExpr(*s.e1);
        emitCondBr(c, body, exit);
        place(body);
        lowerBlock(s.body);
        if (!terminated()) emitBr(hdr);
        place(exit);
        break;
      }
      case SK::Block: lowerBlock(s.body); break;
      case SK::Return: emit(mk(Op::Ret, VT::S32)); break;
    }
  }

  int emitAddr(int base, int idx) {
    int addr = newReg(VT::U64, false);
    Inst a = mk(Op::Addr, VT::U64);
    a.dst = addr; a.a = base; a.b = idx; a.ii = 4;
    emit(a);
    return addr;
  }

  int emitBin(IBin op, VT ty, int a, int b) {
    int r = newReg(ty, false);
    Inst i = mk(Op::Bin, ty);
    i.dst = r; i.a = a; i.b = b; i.sub = static_cast<int>(op);
    emit(i);
    return r;
  }
  int emitSpecial(Spec s) {
    int r = newReg(VT::S32, false);
    Inst i = mk(Op::Special, VT::S32);
    i.dst = r; i.sub = static_cast<int>(s);
    emit(i);
    return r;
  }

  int lowerShortCircuit(const Expr& e) {
    bool isAnd = e.bop == BinOp::And;
    int res = newReg(VT::Pred, true);
    int lhs = lowerExpr(*e.args[0]);
    assign(res, lhs);
    int rhsB = create(), endB = create();
    if (isAnd) emitCondBr(res, rhsB, endB);
    else emitCondBr(res, endB, rhsB);
    place(rhsB);
    int rhs = lowerExpr(*e.args[1]);
    assign(res, rhs);
    emitBr(endB);
    place(endB);
    return res;
  }

  int lowerExpr(const Expr& e) {
    switch (e.kind) {
      case EK::IntLit: {
        int r = newReg(VT::S32, false);
        Inst c = mk(Op::Const, VT::S32);
        c.dst = r; c.ii = e.ival;
        emit(c);
        return r;
      }
      case EK::FloatLit: {
        int r = newReg(VT::F32, false);
        Inst c = mk(Op::Const, VT::F32);
        c.dst = r; c.ff = e.fval;
        emit(c);
        return r;
      }
      case EK::Var: return lookup(e.name);
      case EK::Unary: {
        int a = lowerExpr(*e.args[0]);
        if (e.uop == UnOp::Neg) {
          int r = newReg(vtOf(e.ty), false);
          Inst i = mk(Op::Neg, vtOf(e.ty));
          i.dst = r; i.a = a;
          emit(i);
          return r;
        }
        int r = newReg(VT::Pred, false);
        Inst i = mk(Op::Not, VT::Pred);
        i.dst = r; i.a = a;
        emit(i);
        return r;
      }
      case EK::Binary: {
        if (e.bop == BinOp::And || e.bop == BinOp::Or) return lowerShortCircuit(e);
        int a = lowerExpr(*e.args[0]);
        int b = lowerExpr(*e.args[1]);
        switch (e.bop) {
          case BinOp::Lt: case BinOp::Le: case BinOp::Gt: case BinOp::Ge: case BinOp::Eq: case BinOp::Ne: {
            static const ICmp map[] = {ICmp::Lt, ICmp::Le, ICmp::Gt, ICmp::Ge, ICmp::Eq, ICmp::Ne};
            int idx = static_cast<int>(e.bop) - static_cast<int>(BinOp::Lt);
            int r = newReg(VT::Pred, false);
            Inst i = mk(Op::Cmp, vtOf(e.args[0]->ty));
            i.dst = r; i.a = a; i.b = b; i.sub = static_cast<int>(map[idx]);
            emit(i);
            return r;
          }
          case BinOp::Add: return emitBin(IBin::Add, vtOf(e.ty), a, b);
          case BinOp::Sub: return emitBin(IBin::Sub, vtOf(e.ty), a, b);
          case BinOp::Mul: return emitBin(IBin::Mul, vtOf(e.ty), a, b);
          case BinOp::Div: return emitBin(IBin::Div, vtOf(e.ty), a, b);
          default: return emitBin(IBin::Rem, vtOf(e.ty), a, b);
        }
      }
      case EK::Index: {
        int base = lookup(e.args[0]->name);
        int idx = lowerExpr(*e.args[1]);
        int addr = emitAddr(base, idx);
        int r = newReg(vtOf(e.ty), false);
        Inst i = mk(Op::Load, vtOf(e.ty));
        i.dst = r; i.a = addr;
        emit(i);
        return r;
      }
      case EK::Call: {
        const std::string& n = e.name;
        if (n == "threadIdx") return emitSpecial(Spec::ThreadIdx);
        if (n == "blockIdx") return emitSpecial(Spec::BlockIdx);
        if (n == "blockDim") return emitSpecial(Spec::BlockDim);
        if (n == "gid") {  // blockIdx * blockDim + threadIdx
          int bi = emitSpecial(Spec::BlockIdx);
          int bd = emitSpecial(Spec::BlockDim);
          int ti = emitSpecial(Spec::ThreadIdx);
          int m = emitBin(IBin::Mul, VT::S32, bi, bd);
          return emitBin(IBin::Add, VT::S32, m, ti);
        }
        if (n == "sqrt") {
          int a = lowerExpr(*e.args[0]);
          int r = newReg(VT::F32, false);
          Inst i = mk(Op::Sqrt, VT::F32);
          i.dst = r; i.a = a;
          emit(i);
          return r;
        }
        int a = lowerExpr(*e.args[0]);
        int b = lowerExpr(*e.args[1]);
        return emitBin(n == "min" ? IBin::Min : IBin::Max, vtOf(e.ty), a, b);
      }
      case EK::Cast: {
        int a = lowerExpr(*e.args[0]);
        if (e.args[0]->ty == e.ty) return a;
        int r = newReg(vtOf(e.ty), false);
        Inst i = mk(Op::Cast, vtOf(e.ty));
        i.dst = r; i.a = a;
        i.sub = static_cast<int>(e.ty == tFloat() ? CastK::I2F : CastK::F2I);
        emit(i);
        return r;
      }
    }
    return -1;
  }
};

}  // namespace ptxlite
