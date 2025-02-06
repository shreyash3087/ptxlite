// Abstract syntax tree.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"

namespace ptxlite {

enum class BinOp { Add, Sub, Mul, Div, Rem, Lt, Le, Gt, Ge, Eq, Ne, And, Or };
enum class UnOp { Neg, Not };
enum class EK { IntLit, FloatLit, Var, Unary, Binary, Index, Call, Cast };

struct Expr;
using ExprP = std::unique_ptr<Expr>;

struct Expr {
  EK kind = EK::IntLit;
  int line = 0, col = 0;
  Type ty;                  // filled in by Sema
  int32_t ival = 0;
  float fval = 0;
  std::string name;         // Var / Call
  BinOp bop = BinOp::Add;
  UnOp uop = UnOp::Neg;
  std::vector<ExprP> args;  // Unary:[x] Binary:[l,r] Index:[base,idx] Call:[args] Cast:[x]
};

inline ExprP cloneExpr(const Expr& e) {
  auto c = std::make_unique<Expr>();
  c->kind = e.kind; c->line = e.line; c->col = e.col; c->ty = e.ty;
  c->ival = e.ival; c->fval = e.fval; c->name = e.name; c->bop = e.bop; c->uop = e.uop;
  for (auto& a : e.args) c->args.push_back(cloneExpr(*a));
  return c;
}

enum class SK { Decl, Assign, Store, If, While, Block, Return };

struct Stmt;
using StmtP = std::unique_ptr<Stmt>;

struct Stmt {
  SK kind = SK::Block;
  int line = 0, col = 0;
  std::string name;
  Type declTy;
  ExprP e1, e2;  // Decl:e1=init  Assign:e1=value  Store:e1=index,e2=value  If/While:e1=cond
  std::vector<StmtP> body, other;  // If: body=then, other=else; While/Block: body
};

struct Param { std::string name; Type ty; int line = 0, col = 0; };

struct Kernel {
  std::string name;
  std::vector<Param> params;
  std::vector<StmtP> body;
  int line = 0, col = 0;
};

struct Program { std::vector<Kernel> kernels; };

}  // namespace ptxlite
