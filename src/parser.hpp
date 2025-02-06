// Recursive-descent parser with precedence climbing for expressions.
//
// Grammar (informal):
//   program  := kernel+
//   kernel   := 'kernel' IDENT '(' [param (',' param)*] ')' block
//   param    := ('int' | 'float') ['*'] IDENT
//   stmt     := block | decl ';' | assign ';' | store ';' | 'return' ';'
//             | 'if' '(' expr ')' stmt ['else' stmt] | 'while' '(' expr ')' stmt
//             | 'for' '(' [simple] ';' expr ';' [simple] ')' stmt
//   expr     := || && == != < <= > >= + - * / % unary(- !) postfix(index, call, cast)
#pragma once
#include <algorithm>

#include "ast.hpp"
#include "lexer.hpp"

namespace ptxlite {

class Parser {
 public:
  explicit Parser(std::vector<Token> toks) : t_(std::move(toks)) {}

  Program parseProgram() {
    Program p;
    while (peek().kind != Tok::End) p.kernels.push_back(parseKernel());
    if (p.kernels.empty()) errAt(peek(), "expected at least one kernel");
    return p;
  }

 private:
  std::vector<Token> t_;
  size_t i_ = 0;

  const Token& peek(size_t k = 0) const { return t_[std::min(i_ + k, t_.size() - 1)]; }
  Token take() {
    Token t = t_[i_];
    if (i_ + 1 < t_.size()) ++i_;
    return t;
  }
  bool accept(Tok k) {
    if (peek().kind == k) { take(); return true; }
    return false;
  }
  Token expect(Tok k, const char* what) {
    if (peek().kind != k) errAt(peek(), std::string("expected ") + what);
    return take();
  }
  [[noreturn]] void errAt(const Token& t, const std::string& m) { throw CompileError(m, t.line, t.col); }

  static StmtP mkStmt(SK k, const Token& at) {
    auto s = std::make_unique<Stmt>();
    s->kind = k; s->line = at.line; s->col = at.col;
    return s;
  }
  static ExprP mkExpr(EK k, const Token& at) {
    auto e = std::make_unique<Expr>();
    e->kind = k; e->line = at.line; e->col = at.col;
    return e;
  }

  Kernel parseKernel() {
    Kernel k;
    Token kw = expect(Tok::KwKernel, "'kernel'");
    k.line = kw.line; k.col = kw.col;
    k.name = expect(Tok::Ident, "kernel name").text;
    expect(Tok::LParen, "'('");
    if (peek().kind != Tok::RParen) {
      do { k.params.push_back(parseParam()); } while (accept(Tok::Comma));
    }
    expect(Tok::RParen, "')'");
    expect(Tok::LBrace, "'{'");
    k.body = parseBlockBody();
    return k;
  }

  Param parseParam() {
    Param p;
    Token t = peek();
    if (t.kind == Tok::KwInt) p.ty.base = Base::Int;
    else if (t.kind == Tok::KwFloat) p.ty.base = Base::Float;
    else errAt(t, "expected parameter type ('int' or 'float')");
    take();
    p.ty.ptr = accept(Tok::Star);
    Token n = expect(Tok::Ident, "parameter name");
    p.name = n.text; p.line = n.line; p.col = n.col;
    return p;
  }

  std::vector<StmtP> parseBlockBody() {
    std::vector<StmtP> out;
    while (peek().kind != Tok::RBrace) {
      if (peek().kind == Tok::End) errAt(peek(), "unexpected end of file, expected '}'");
      out.push_back(parseStmt());
    }
    take();
    return out;
  }

  std::vector<StmtP> parseStmtAsList() {
    StmtP s = parseStmt();
    if (s->kind == SK::Block) return std::move(s->body);
    std::vector<StmtP> v;
    v.push_back(std::move(s));
    return v;
  }

  StmtP parseStmt() {
    Token t = peek();
    switch (t.kind) {
      case Tok::LBrace: {
        take();
        auto s = mkStmt(SK::Block, t);
        s->body = parseBlockBody();
        return s;
      }
      case Tok::KwIf: {
        take();
        auto s = mkStmt(SK::If, t);
        expect(Tok::LParen, "'(' after 'if'");
        s->e1 = parseExpr();
        expect(Tok::RParen, "')'");
        s->body = parseStmtAsList();
        if (accept(Tok::KwElse)) s->other = parseStmtAsList();
        return s;
      }
      case Tok::KwWhile: {
        take();
        auto s = mkStmt(SK::While, t);
        expect(Tok::LParen, "'(' after 'while'");
        s->e1 = parseExpr();
        expect(Tok::RParen, "')'");
        s->body = parseStmtAsList();
        return s;
      }
      case Tok::KwFor: return parseFor();
      case Tok::KwReturn: {
        take();
        expect(Tok::Semi, "';' after 'return' (kernels return void)");
        return mkStmt(SK::Return, t);
      }
      default: {
        auto s = parseSimple();
        expect(Tok::Semi, "';'");
        return s;
      }
    }
  }

  // for (init; cond; step) body   ==>   { init; while (cond) { body; step; } }
  StmtP parseFor() {
    Token t = take();
    expect(Tok::LParen, "'(' after 'for'");
    StmtP init;
    if (peek().kind != Tok::Semi) init = parseSimple();
    expect(Tok::Semi, "';'");
    if (peek().kind == Tok::Semi) errAt(peek(), "expected a loop condition");
    ExprP cond = parseExpr();
    expect(Tok::Semi, "';'");
    StmtP step;
    if (peek().kind != Tok::RParen) step = parseSimple();
    expect(Tok::RParen, "')'");
    auto w = mkStmt(SK::While, t);
    w->e1 = std::move(cond);
    w->body = parseStmtAsList();
    if (step) w->body.push_back(std::move(step));
    if (!init) return w;
    auto blk = mkStmt(SK::Block, t);
    blk->body.push_back(std::move(init));
    blk->body.push_back(std::move(w));
    return blk;
  }

  static bool isAssignOp(Tok k) {
    return k == Tok::Assign || k == Tok::PlusEq || k == Tok::MinusEq || k == Tok::StarEq || k == Tok::SlashEq;
  }
  static BinOp compoundOp(Tok k) {
    switch (k) {
      case Tok::PlusEq: return BinOp::Add;
      case Tok::MinusEq: return BinOp::Sub;
      case Tok::StarEq: return BinOp::Mul;
      default: return BinOp::Div;
    }
  }
  static ExprP mkBinary(BinOp op, ExprP l, ExprP r, const Token& at) {
    auto e = mkExpr(EK::Binary, at);
    e->bop = op;
    e->args.push_back(std::move(l));
    e->args.push_back(std::move(r));
    return e;
  }

  // Declaration, assignment, compound assignment, or indexed store (no trailing ';').
  StmtP parseSimple() {
    Token t = peek();
    if (t.kind == Tok::KwInt || t.kind == Tok::KwFloat) {
      take();
      if (peek().kind == Tok::Star) errAt(peek(), "local pointer variables are not supported");
      Token n = expect(Tok::Ident, "variable name");
      auto s = mkStmt(SK::Decl, t);
      s->name = n.text;
      s->declTy = t.kind == Tok::KwInt ? tInt() : tFloat();
      if (accept(Tok::Assign)) s->e1 = parseExpr();
      return s;
    }
    if (t.kind == Tok::Ident) {
      if (peek(1).kind == Tok::LBracket) {
        Token name = take();
        take();
        ExprP idx = parseExpr();
        expect(Tok::RBracket, "']'");
        if (!isAssignOp(peek().kind)) errAt(peek(), "expected '=' after indexed expression");
        Token op = take();
        ExprP rhs = parseExpr();
        auto s = mkStmt(SK::Store, t);
        s->name = name.text;
        if (op.kind != Tok::Assign) {
          auto base = mkExpr(EK::Var, name);
          base->name = name.text;
          auto load = mkExpr(EK::Index, name);
          load->args.push_back(std::move(base));
          load->args.push_back(cloneExpr(*idx));
          rhs = mkBinary(compoundOp(op.kind), std::move(load), std::move(rhs), op);
        }
        s->e1 = std::move(idx);
        s->e2 = std::move(rhs);
        return s;
      }
      if (isAssignOp(peek(1).kind)) {
        Token name = take();
        Token op = take();
        ExprP rhs = parseExpr();
        auto s = mkStmt(SK::Assign, t);
        s->name = name.text;
        if (op.kind != Tok::Assign) {
          auto v = mkExpr(EK::Var, name);
          v->name = name.text;
          rhs = mkBinary(compoundOp(op.kind), std::move(v), std::move(rhs), op);
        }
        s->e1 = std::move(rhs);
        return s;
      }
    }
    errAt(t, "expected a statement");
  }

  static int prec(Tok k) {
    switch (k) {
      case Tok::OrOr: return 1;
      case Tok::AndAnd: return 2;
      case Tok::EqEq: case Tok::Ne: return 3;
      case Tok::Lt: case Tok::Le: case Tok::Gt: case Tok::Ge: return 4;
      case Tok::Plus: case Tok::Minus: return 5;
      case Tok::Star: case Tok::Slash: case Tok::Percent: return 6;
      default: return 0;
    }
  }
  static BinOp binOpOf(Tok k) {
    switch (k) {
      case Tok::OrOr: return BinOp::Or;
      case Tok::AndAnd: return BinOp::And;
      case Tok::EqEq: return BinOp::Eq;
      case Tok::Ne: return BinOp::Ne;
      case Tok::Lt: return BinOp::Lt;
      case Tok::Le: return BinOp::Le;
      case Tok::Gt: return BinOp::Gt;
      case Tok::Ge: return BinOp::Ge;
      case Tok::Plus: return BinOp::Add;
      case Tok::Minus: return BinOp::Sub;
      case Tok::Star: return BinOp::Mul;
      case Tok::Slash: return BinOp::Div;
      default: return BinOp::Rem;
    }
  }

  ExprP parseExpr() { return parseBin(1); }

  ExprP parseBin(int minPrec) {
    ExprP lhs = parseUnary();
    for (;;) {
      int p = prec(peek().kind);
      if (p == 0 || p < minPrec) break;
      Token op = take();
      ExprP rhs = parseBin(p + 1);
      lhs = mkBinary(binOpOf(op.kind), std::move(lhs), std::move(rhs), op);
    }
    return lhs;
  }

  ExprP parseUnary() {
    Token t = peek();
    if (t.kind == Tok::Minus || t.kind == Tok::Bang) {
      take();
      auto e = mkExpr(EK::Unary, t);
      e->uop = t.kind == Tok::Minus ? UnOp::Neg : UnOp::Not;
      e->args.push_back(parseUnary());
      return e;
    }
    return parsePrimary();
  }

  ExprP parsePrimary() {
    Token t = peek();
    switch (t.kind) {
      case Tok::IntLit: {
        take();
        auto e = mkExpr(EK::IntLit, t);
        e->ival = static_cast<int32_t>(t.ival);
        return e;
      }
      case Tok::FloatLit: {
        take();
        auto e = mkExpr(EK::FloatLit, t);
        e->fval = t.fval;
        return e;
      }
      case Tok::LParen: {
        take();
        ExprP e = parseExpr();
        expect(Tok::RParen, "')'");
        return e;
      }
      case Tok::KwInt:
      case Tok::KwFloat: {
        take();
        expect(Tok::LParen, "'(' in cast");
        auto e = mkExpr(EK::Cast, t);
        e->ty = t.kind == Tok::KwInt ? tInt() : tFloat();
        e->args.push_back(parseExpr());
        expect(Tok::RParen, "')'");
        return e;
      }
      case Tok::Ident: {
        take();
        if (peek().kind == Tok::LParen) {
          take();
          auto e = mkExpr(EK::Call, t);
          e->name = t.text;
          if (peek().kind != Tok::RParen) {
            do { e->args.push_back(parseExpr()); } while (accept(Tok::Comma));
          }
          expect(Tok::RParen, "')'");
          return e;
        }
        auto v = mkExpr(EK::Var, t);
        v->name = t.text;
        if (peek().kind == Tok::LBracket) {
          take();
          auto e = mkExpr(EK::Index, t);
          e->args.push_back(std::move(v));
          e->args.push_back(parseExpr());
          expect(Tok::RBracket, "']'");
          return e;
        }
        return v;
      }
      default:
        errAt(t, "expected an expression");
    }
  }
};

}  // namespace ptxlite
