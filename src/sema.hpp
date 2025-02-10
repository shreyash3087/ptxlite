// Semantic analysis: scoping, type checking, and insertion of implicit int->float casts.
#pragma once
#include <map>
#include <set>

#include "ast.hpp"

namespace ptxlite {

class Sema {
 public:
  void checkProgram(Program& p) {
    std::set<std::string> seen;
    for (auto& k : p.kernels) {
      if (!seen.insert(k.name).second)
        throw CompileError("redefinition of kernel '" + k.name + "'", k.line, k.col);
      check(k);
    }
  }

  void check(Kernel& k) {
    scopes_.clear();
    scopes_.emplace_back();
    for (auto& p : k.params) declare(p.name, p.ty, p.line, p.col);
    checkBlock(k.body);
  }

 private:
  std::vector<std::map<std::string, Type>> scopes_;

  void declare(const std::string& n, Type t, int line, int col) {
    if (scopes_.back().count(n)) throw CompileError("redefinition of '" + n + "'", line, col);
    scopes_.back()[n] = t;
  }
  const Type* lookup(const std::string& n) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      auto f = it->find(n);
      if (f != it->end()) return &f->second;
    }
    return nullptr;
  }

  void checkBlock(std::vector<StmtP>& b) {
    scopes_.emplace_back();
    for (auto& s : b) checkStmt(*s);
    scopes_.pop_back();
  }

  static void wrapCast(ExprP& e, Type to) {
    auto c = std::make_unique<Expr>();
    c->kind = EK::Cast; c->line = e->line; c->col = e->col; c->ty = to;
    c->args.push_back(std::move(e));
    e = std::move(c);
  }

  // Implicit conversion: only int -> float is allowed.
  void coerce(ExprP& e, Type to, const char* what) {
    if (e->ty == to) return;
    if (e->ty == tInt() && to == tFloat()) { wrapCast(e, to); return; }
    if (e->ty == tFloat() && to == tInt())
      throw CompileError(std::string("cannot implicitly convert float to int in ") + what + " (use int(...))", e->line, e->col);
    throw CompileError(std::string("cannot convert ") + e->ty.str() + " to " + to.str() + " in " + what, e->line, e->col);
  }

  void checkStmt(Stmt& s) {
    switch (s.kind) {
      case SK::Decl:
        if (s.e1) { checkExpr(s.e1); coerce(s.e1, s.declTy, "initializer"); }
        declare(s.name, s.declTy, s.line, s.col);
        break;
      case SK::Assign: {
        const Type* t = lookup(s.name);
        if (!t) throw CompileError("use of undeclared identifier '" + s.name + "'", s.line, s.col);
        if (t->ptr) throw CompileError("cannot assign to pointer '" + s.name + "'; use an index", s.line, s.col);
        checkExpr(s.e1);
        coerce(s.e1, *t, "assignment");
        break;
      }
      case SK::Store: {
        const Type* t = lookup(s.name);
        if (!t) throw CompileError("use of undeclared identifier '" + s.name + "'", s.line, s.col);
        if (!t->ptr) throw CompileError("'" + s.name + "' is not a pointer and cannot be indexed", s.line, s.col);
        checkExpr(s.e1);
        if (s.e1->ty != tInt()) throw CompileError("array index must be int, got " + s.e1->ty.str(), s.e1->line, s.e1->col);
        checkExpr(s.e2);
        coerce(s.e2, Type{t->base, false}, "store");
        break;
      }
      case SK::If:
        checkExpr(s.e1);
        if (s.e1->ty != tBool()) throw CompileError("condition must be a comparison, got " + s.e1->ty.str(), s.e1->line, s.e1->col);
        checkBlock(s.body);
        checkBlock(s.other);
        break;
      case SK::While:
        checkExpr(s.e1);
        if (s.e1->ty != tBool()) throw CompileError("condition must be a comparison, got " + s.e1->ty.str(), s.e1->line, s.e1->col);
        checkBlock(s.body);
        break;
      case SK::Block: checkBlock(s.body); break;
      case SK::Return: break;
    }
  }

  void unify(ExprP& l, ExprP& r) {
    if (l->ty == r->ty) return;
    if (l->ty == tInt()) wrapCast(l, tFloat());
    else wrapCast(r, tFloat());
  }

  void checkExpr(ExprP& e) {
    switch (e->kind) {
      case EK::IntLit: e->ty = tInt(); break;
      case EK::FloatLit: e->ty = tFloat(); break;
      case EK::Var: {
        const Type* t = lookup(e->name);
        if (!t) throw CompileError("use of undeclared identifier '" + e->name + "'", e->line, e->col);
        if (t->ptr) throw CompileError("pointer '" + e->name + "' can only be used with an index", e->line, e->col);
        e->ty = *t;
        break;
      }
      case EK::Unary: {
        checkExpr(e->args[0]);
        Type t = e->args[0]->ty;
        if (e->uop == UnOp::Neg) {
          if (!t.numeric()) throw CompileError("unary '-' needs a numeric operand, got " + t.str(), e->line, e->col);
          e->ty = t;
        } else {
          if (t != tBool()) throw CompileError("'!' needs a comparison operand, got " + t.str(), e->line, e->col);
          e->ty = tBool();
        }
        break;
      }
      case EK::Binary: checkBinary(*e); break;
      case EK::Index: {
        Expr& base = *e->args[0];
        const Type* t = lookup(base.name);
        if (!t) throw CompileError("use of undeclared identifier '" + base.name + "'", base.line, base.col);
        if (!t->ptr) throw CompileError("'" + base.name + "' is not a pointer and cannot be indexed", base.line, base.col);
        base.ty = *t;
        checkExpr(e->args[1]);
        if (e->args[1]->ty != tInt())
          throw CompileError("array index must be int, got " + e->args[1]->ty.str(), e->args[1]->line, e->args[1]->col);
        e->ty = Type{t->base, false};
        break;
      }
      case EK::Call: checkCall(*e); break;
      case EK::Cast: {
        checkExpr(e->args[0]);
        if (!e->args[0]->ty.numeric())
          throw CompileError("cannot cast " + e->args[0]->ty.str() + " to " + e->ty.str(), e->line, e->col);
        break;
      }
    }
  }

  void checkBinary(Expr& e) {
    checkExpr(e.args[0]);
    checkExpr(e.args[1]);
    ExprP& l = e.args[0];
    ExprP& r = e.args[1];
    auto bad = [&](const char* need) {
      throw CompileError(std::string("operator needs ") + need + " operands, got " + l->ty.str() + " and " + r->ty.str(), e.line, e.col);
    };
    switch (e.bop) {
      case BinOp::And: case BinOp::Or:
        if (l->ty != tBool() || r->ty != tBool()) bad("comparison (bool)");
        e.ty = tBool();
        break;
      case BinOp::Rem:
        if (l->ty != tInt() || r->ty != tInt()) bad("int");
        e.ty = tInt();
        break;
      case BinOp::Add: case BinOp::Sub: case BinOp::Mul: case BinOp::Div:
        if (!l->ty.numeric() || !r->ty.numeric()) bad("numeric");
        unify(l, r);
        e.ty = l->ty;
        break;
      default:  // comparisons
        if (!l->ty.numeric() || !r->ty.numeric()) bad("numeric");
        unify(l, r);
        e.ty = tBool();
        break;
    }
  }

  void checkCall(Expr& e) {
    const std::string& n = e.name;
    auto argc = [&](size_t k) {
      if (e.args.size() != k)
        throw CompileError("'" + n + "' expects " + std::to_string(k) + " argument(s), got " + std::to_string(e.args.size()), e.line, e.col);
    };
    if (n == "threadIdx" || n == "blockIdx" || n == "blockDim" || n == "gid") {
      argc(0);
      e.ty = tInt();
    } else if (n == "sqrt") {
      argc(1);
      checkExpr(e.args[0]);
      coerce(e.args[0], tFloat(), "argument of sqrt");
      e.ty = tFloat();
    } else if (n == "min" || n == "max") {
      argc(2);
      checkExpr(e.args[0]);
      checkExpr(e.args[1]);
      if (!e.args[0]->ty.numeric() || !e.args[1]->ty.numeric())
        throw CompileError("'" + n + "' needs numeric arguments", e.line, e.col);
      unify(e.args[0], e.args[1]);
      e.ty = e.args[0]->ty;
    } else {
      throw CompileError("call to unknown function '" + n + "'", e.line, e.col);
    }
  }
};

}  // namespace ptxlite
