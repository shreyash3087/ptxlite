// Hand-written lexer for the kernel language.
#pragma once
#include <climits>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"

namespace ptxlite {

enum class Tok {
  End, Ident, IntLit, FloatLit,
  KwKernel, KwInt, KwFloat, KwIf, KwElse, KwWhile, KwFor, KwReturn,
  LParen, RParen, LBrace, RBrace, LBracket, RBracket, Semi, Comma,
  Assign, PlusEq, MinusEq, StarEq, SlashEq,
  Plus, Minus, Star, Slash, Percent, Bang,
  Lt, Le, Gt, Ge, EqEq, Ne, AndAnd, OrOr
};

struct Token {
  Tok kind = Tok::End;
  std::string text;
  long long ival = 0;
  float fval = 0;
  int line = 1, col = 1;
};

class Lexer {
 public:
  explicit Lexer(std::string src) : src_(std::move(src)) {}

  std::vector<Token> run() {
    std::vector<Token> out;
    for (;;) {
      Token t = next();
      out.push_back(t);
      if (t.kind == Tok::End) break;
    }
    return out;
  }

 private:
  std::string src_;
  size_t pos_ = 0;
  int line_ = 1, col_ = 1;

  char peek(size_t k = 0) const { return pos_ + k < src_.size() ? src_[pos_ + k] : '\0'; }
  char adv() {
    char c = peek();
    if (c == '\0') return c;
    ++pos_;
    if (c == '\n') { ++line_; col_ = 1; } else { ++col_; }
    return c;
  }

  void skipTrivia() {
    for (;;) {
      char c = peek();
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        adv();
      } else if (c == '/' && peek(1) == '/') {
        while (peek() && peek() != '\n') adv();
      } else if (c == '/' && peek(1) == '*') {
        int l = line_, cl = col_;
        adv(); adv();
        while (peek() && !(peek() == '*' && peek(1) == '/')) adv();
        if (!peek()) throw CompileError("unterminated block comment", l, cl);
        adv(); adv();
      } else {
        break;
      }
    }
  }

  Token next() {
    skipTrivia();
    Token t;
    t.line = line_;
    t.col = col_;
    char c = peek();
    if (c == '\0') { t.kind = Tok::End; return t; }

    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      std::string s;
      while (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_') s += adv();
      static const std::map<std::string, Tok> kw = {
          {"kernel", Tok::KwKernel}, {"int", Tok::KwInt}, {"float", Tok::KwFloat}, {"if", Tok::KwIf},
          {"else", Tok::KwElse},     {"while", Tok::KwWhile}, {"for", Tok::KwFor}, {"return", Tok::KwReturn}};
      auto it = kw.find(s);
      t.kind = it == kw.end() ? Tok::Ident : it->second;
      t.text = s;
      return t;
    }

    if (std::isdigit(static_cast<unsigned char>(c))) return number(t);

    auto two = [&](char a, char b) { return peek() == a && peek(1) == b; };
    if (two('=', '=')) { adv(); adv(); t.kind = Tok::EqEq; return t; }
    if (two('!', '=')) { adv(); adv(); t.kind = Tok::Ne; return t; }
    if (two('<', '=')) { adv(); adv(); t.kind = Tok::Le; return t; }
    if (two('>', '=')) { adv(); adv(); t.kind = Tok::Ge; return t; }
    if (two('&', '&')) { adv(); adv(); t.kind = Tok::AndAnd; return t; }
    if (two('|', '|')) { adv(); adv(); t.kind = Tok::OrOr; return t; }
    if (two('+', '=')) { adv(); adv(); t.kind = Tok::PlusEq; return t; }
    if (two('-', '=')) { adv(); adv(); t.kind = Tok::MinusEq; return t; }
    if (two('*', '=')) { adv(); adv(); t.kind = Tok::StarEq; return t; }
    if (two('/', '=')) { adv(); adv(); t.kind = Tok::SlashEq; return t; }

    adv();
    switch (c) {
      case '(': t.kind = Tok::LParen; break;
      case ')': t.kind = Tok::RParen; break;
      case '{': t.kind = Tok::LBrace; break;
      case '}': t.kind = Tok::RBrace; break;
      case '[': t.kind = Tok::LBracket; break;
      case ']': t.kind = Tok::RBracket; break;
      case ';': t.kind = Tok::Semi; break;
      case ',': t.kind = Tok::Comma; break;
      case '=': t.kind = Tok::Assign; break;
      case '+': t.kind = Tok::Plus; break;
      case '-': t.kind = Tok::Minus; break;
      case '*': t.kind = Tok::Star; break;
      case '/': t.kind = Tok::Slash; break;
      case '%': t.kind = Tok::Percent; break;
      case '!': t.kind = Tok::Bang; break;
      case '<': t.kind = Tok::Lt; break;
      case '>': t.kind = Tok::Gt; break;
      default:
        throw CompileError(std::string("unexpected character '") + c + "'", t.line, t.col);
    }
    return t;
  }

  Token number(Token t) {
    std::string s;
    bool isFloat = false;
    while (std::isdigit(static_cast<unsigned char>(peek()))) s += adv();
    if (peek() == '.') {
      isFloat = true;
      s += adv();
      while (std::isdigit(static_cast<unsigned char>(peek()))) s += adv();
    }
    if (peek() == 'e' || peek() == 'E') {
      char sign = peek(1);
      size_t d = (sign == '+' || sign == '-') ? 2 : 1;
      if (std::isdigit(static_cast<unsigned char>(peek(d)))) {
        isFloat = true;
        s += adv();
        if (d == 2) s += adv();
        while (std::isdigit(static_cast<unsigned char>(peek()))) s += adv();
      }
    }
    if (peek() == 'f' || peek() == 'F') { isFloat = true; adv(); }
    t.text = s;
    if (isFloat) {
      t.kind = Tok::FloatLit;
      t.fval = std::strtof(s.c_str(), nullptr);
    } else {
      t.kind = Tok::IntLit;
      t.ival = std::strtoll(s.c_str(), nullptr, 10);
      if (s.size() > 10 || t.ival > INT_MAX)
        throw CompileError("integer literal '" + s + "' does not fit in 32 bits", t.line, t.col);
    }
    return t;
  }
};

}  // namespace ptxlite
