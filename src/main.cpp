// ptxlite CLI: compile a .k kernel source to PTX (or dump IR).
//
// Usage: ptxlite [options] <input.k>
//   -O0 / -O1 / -O2     optimization level (default -O2)
//   --no-fma             disable mul-add fusion
//   --arch=sm_XX         PTX target (default sm_75)
//   --emit-ir            print IR to stdout and exit
//   --emit-ptx           print PTX to stdout (default when -o is absent)
//   -o <file>            write PTX to file
//   --stats              print optimization statistics to stderr
//
// Errors are printed as:   file:line:col: error: message
// followed by the offending source line and a caret.
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "compiler.hpp"

static std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "ptxlite: cannot open '" << path << "'\n";
    std::exit(1);
  }
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

static std::string getLine(const std::string& src, int line) {
  int cur = 1;
  size_t start = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    if (src[i] == '\n') {
      if (cur == line) return src.substr(start, i - start);
      ++cur;
      start = i + 1;
    }
  }
  if (cur == line) return src.substr(start);
  return "";
}

int main(int argc, char** argv) {
  ptxlite::Options opts;
  bool emitIr = false;
  bool emitPtx = true;
  bool printStats = false;
  std::string outFile;
  std::string inFile;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-O0") { opts.opt = 0; }
    else if (a == "-O1") { opts.opt = 1; }
    else if (a == "-O2") { opts.opt = 2; }
    else if (a == "--no-fma") { opts.fma = false; }
    else if (a.substr(0, 7) == "--arch=") { opts.arch = a.substr(7); }
    else if (a == "--emit-ir") { emitIr = true; emitPtx = false; }
    else if (a == "--emit-ptx") { emitPtx = true; }
    else if (a == "--stats") { printStats = true; }
    else if (a == "-o") {
      if (i + 1 >= argc) { std::cerr << "ptxlite: -o requires an argument\n"; return 1; }
      outFile = argv[++i];
    } else if (a[0] == '-') {
      std::cerr << "ptxlite: unknown option '" << a << "'\n";
      return 1;
    } else {
      inFile = a;
    }
  }

  if (inFile.empty()) {
    std::cerr << "Usage: ptxlite [options] <input.k>\n";
    return 1;
  }

  std::string src = readFile(inFile);

  try {
    ptxlite::Stats stats;
    auto funcs = ptxlite::compile(src, opts, &stats);

    if (emitIr) {
      for (auto& f : funcs) std::cout << ptxlite::toString(f);
      return 0;
    }

    std::string ptx = ptxlite::emitPTX(funcs, opts);

    if (!outFile.empty()) {
      std::ofstream out(outFile);
      if (!out.is_open()) { std::cerr << "ptxlite: cannot write '" << outFile << "'\n"; return 1; }
      out << ptx;
    } else if (emitPtx) {
      std::cout << ptx;
    }

    if (printStats) {
      std::cerr << "stats: before=" << stats.instsBefore << " after=" << stats.instsAfter
                << " folded=" << stats.folded << " simplified=" << stats.simplified
                << " strength=" << stats.strength << " copies=" << stats.copies
                << " branches=" << stats.branches << " dead=" << stats.dead
                << " fused=" << stats.fused << "\n";
    }
  } catch (const ptxlite::CompileError& e) {
    std::string line = getLine(src, e.line);
    std::cerr << inFile << ":" << e.line << ":" << e.col << ": error: " << e.what() << "\n";
    if (!line.empty()) {
      std::cerr << line << "\n";
      std::cerr << std::string(e.col > 0 ? e.col - 1 : 0, ' ') << "^\n";
    }
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "ptxlite: internal error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
