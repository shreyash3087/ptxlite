// IR interpreter: simulates the kernel sequentially over a 1-D thread grid.
//
// Pointer encoding: a Value holds either an integer, a float, or an address
// (buffer index + byte offset packed into a uint64_t). Buffers are described by
// the caller as contiguous slices of a flat arena; the interpreter checks every
// load and store for out-of-range access and throws on violation.
//
// Thread model: one block, threads are run sequentially. gid() = threadIdx.
// For multi-block launches the caller loops over blocks and sets blockIdx_.
#pragma once
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ir.hpp"
#include "semantics.hpp"

namespace ptxlite {

// A buffer visible to the interpreter: pointer, element count, element size.
struct Buffer {
  void* data = nullptr;
  size_t count = 0;
  size_t elemSize = 4;  // bytes per element
};

// An encoded pointer: buffer index in the high 32 bits, byte offset in low 32.
static inline uint64_t encPtr(int bufIdx, uint32_t byteOff) {
  return (static_cast<uint64_t>(bufIdx) << 32) | byteOff;
}
static inline int ptrBuf(uint64_t p) { return static_cast<int>(p >> 32); }
static inline uint32_t ptrOff(uint64_t p) { return static_cast<uint32_t>(p); }

union Value {
  int32_t i;
  float f;
  uint64_t p;   // encoded pointer
};

class Interpreter {
 public:
  explicit Interpreter(const std::vector<Buffer>& bufs, int stepLimit = 10000000)
      : bufs_(bufs), stepLimit_(stepLimit) {}

  // Launch kernel over (numBlocks * blockDim) threads.
  void launch(const Func& fn, const std::vector<Value>& args,
               int numBlocks, int blockDim) {
    for (int bi = 0; bi < numBlocks; ++bi)
      for (int ti = 0; ti < blockDim; ++ti)
        runThread(fn, args, bi, ti, blockDim);
  }

 private:
  const std::vector<Buffer>& bufs_;
  int stepLimit_;

  void runThread(const Func& fn, const std::vector<Value>& args,
                 int blockIdx, int threadIdx, int blockDim) {
    std::vector<Value> reg(fn.regTy.size());
    int steps = 0;

    // Execute from entry block
    int curId = fn.blocks[0].id;
    for (;;) {
      int idx = fn.indexOf(curId);
      if (idx < 0) break;
      const Block& blk = fn.blocks[idx];
      int nextId = -1;
      for (const Inst& in : blk.insts) {
        if (++steps > stepLimit_)
          throw std::runtime_error("interpreter: step limit exceeded (possible infinite loop)");
        nextId = execInst(in, reg, fn, args, blockIdx, threadIdx, blockDim);
        if (nextId == -2) return;  // Ret
        if (nextId >= 0) break;   // branch
      }
      if (nextId < 0) break;  // no explicit branch, fall off end (shouldn't happen in well-formed IR)
      curId = nextId;
    }
  }

  // Returns >= 0 for a branch target block id, -2 for Ret, -1 for continue.
  int execInst(const Inst& in, std::vector<Value>& reg, const Func& fn,
               const std::vector<Value>& args,
               int blockIdx, int threadIdx, int blockDim) {
    Value a = in.a >= 0 ? reg[in.a] : Value{};
    Value b = in.b >= 0 ? reg[in.b] : Value{};
    Value c = in.c >= 0 ? reg[in.c] : Value{};

    switch (in.op) {
      case Op::Const:
        if (in.ty == VT::F32) reg[in.dst].f = in.ff;
        else reg[in.dst].i = in.ii;
        break;
      case Op::Mov:
        reg[in.dst] = a;
        break;
      case Op::Param:
        reg[in.dst] = args[in.ii];
        break;
      case Op::Special:
        switch (static_cast<Spec>(in.sub)) {
          case Spec::ThreadIdx: reg[in.dst].i = threadIdx; break;
          case Spec::BlockIdx:  reg[in.dst].i = blockIdx;  break;
          case Spec::BlockDim:  reg[in.dst].i = blockDim;  break;
        }
        break;
      case Op::Bin: {
        IBin op = static_cast<IBin>(in.sub);
        if (fn.regTy[in.dst] == VT::F32)
          reg[in.dst].f = evalF(op, a.f, b.f);
        else
          reg[in.dst].i = evalI(op, a.i, b.i);
        break;
      }
      case Op::Cmp: {
        ICmp cmp = static_cast<ICmp>(in.sub);
        bool r = in.ty == VT::F32 ? evalCmpF(cmp, a.f, b.f) : evalCmpI(cmp, a.i, b.i);
        reg[in.dst].i = r ? 1 : 0;
        break;
      }
      case Op::Neg:
        if (fn.regTy[in.a] == VT::F32) reg[in.dst].f = -a.f;
        else reg[in.dst].i = static_cast<int32_t>(0u - static_cast<uint32_t>(a.i));
        break;
      case Op::Not:
        reg[in.dst].i = a.i ? 0 : 1;
        break;
      case Op::Cast:
        if (in.sub == static_cast<int>(CastK::I2F))
          reg[in.dst].f = static_cast<float>(a.i);
        else
          reg[in.dst].i = f2i(a.f);
        break;
      case Op::Addr: {
        // base is a pointer (encoded), idx is an integer, elem size is in.ii
        uint64_t base = reg[in.a].p;
        int32_t idx = reg[in.b].i;
        uint32_t newOff = ptrOff(base) + static_cast<uint32_t>(idx * in.ii);
        reg[in.dst].p = encPtr(ptrBuf(base), newOff);
        break;
      }
      case Op::Load: {
        uint64_t ptr = reg[in.a].p;
        int bi = ptrBuf(ptr);
        uint32_t off = ptrOff(ptr);
        boundsCheck(bi, off, 4);
        const Buffer& buf = bufs_[bi];
        if (fn.regTy[in.dst] == VT::F32)
          std::memcpy(&reg[in.dst].f, static_cast<char*>(buf.data) + off, 4);
        else
          std::memcpy(&reg[in.dst].i, static_cast<char*>(buf.data) + off, 4);
        break;
      }
      case Op::Store: {
        uint64_t ptr = reg[in.a].p;
        int bi = ptrBuf(ptr);
        uint32_t off = ptrOff(ptr);
        boundsCheck(bi, off, 4);
        const Buffer& buf = bufs_[bi];
        if (in.ty == VT::F32)
          std::memcpy(static_cast<char*>(buf.data) + off, &reg[in.b].f, 4);
        else
          std::memcpy(static_cast<char*>(buf.data) + off, &reg[in.b].i, 4);
        break;
      }
      case Op::Sqrt:
        reg[in.dst].f = std::sqrt(a.f);
        break;
      case Op::MulAdd:
        if (fn.regTy[in.dst] == VT::F32)
          reg[in.dst].f = evalMulAddF(a.f, b.f, c.f);
        else
          reg[in.dst].i = evalMulAddI(a.i, b.i, c.i);
        break;
      case Op::Br:
        return in.t1;
      case Op::CondBr:
        return reg[in.a].i ? in.t1 : in.t2;
      case Op::Ret:
        return -2;
    }
    return -1;
  }

  void boundsCheck(int bufIdx, uint32_t off, size_t sz) const {
    if (bufIdx < 0 || bufIdx >= static_cast<int>(bufs_.size()))
      throw std::runtime_error("interpreter: invalid buffer index " + std::to_string(bufIdx));
    const Buffer& buf = bufs_[bufIdx];
    size_t limit = buf.count * buf.elemSize;
    if (off + sz > limit)
      throw std::runtime_error("interpreter: out-of-bounds access at offset " +
                               std::to_string(off) + " in buffer " + std::to_string(bufIdx) +
                               " (size " + std::to_string(limit) + ")");
  }
};

}  // namespace ptxlite
