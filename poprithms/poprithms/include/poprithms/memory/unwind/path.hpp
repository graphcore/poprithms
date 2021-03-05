// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_PATH_HPP
#define POPRITHMS_MEMORY_UNWIND_PATH_HPP

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/memory/unwind/valuedtensorid.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using common::multiout::InIndex;
using memory::nest::DisjointRegions;

/**
 * An Op, an input and output index, and a direction.
 * */
class Link {
public:
  Link(OpId opId, InIndex inIndex, OutIndex outIndex, bool isFwd)
      : opId_(opId), inIndex_(inIndex), outIndex_(outIndex), isFwd_(isFwd) {}

  static Link fwd(OpId opId, InIndex inIndex, OutIndex outIndex) {
    return Link(opId, inIndex, outIndex, true);
  }
  static Link bwd(OpId opId, InIndex inIndex, OutIndex outIndex) {
    return Link(opId, inIndex, outIndex, false);
  }

  OpId opId() const { return opId_; }
  InIndex inIndex() const { return inIndex_; }
  OutIndex outIndex() const { return outIndex_; }
  bool isFwd() const { return isFwd_; }
  void append(std::ostream &) const;

  Link reverse() const {
    return Link(opId(), inIndex(), outIndex(), !isFwd());
  }

private:
  OpId opId_;
  InIndex inIndex_;
  OutIndex outIndex_;
  bool isFwd_;
};
using Links = std::vector<Link>;
std::ostream &operator<<(std::ostream &, const Link &);
std::ostream &operator<<(std::ostream &, const Links &);

/**
 * A starting Tensor, an ending Tensor, and a Chain connection them.
 * */
class Path {
public:
  Path(const TensorId &src, const chain::Chain &, const TensorId &dst);

  TensorId src() const { return src_; }
  TensorId dst() const { return dst_; }

  /** The Region in the output which the full input gets mapped to through the
   * Chain. */
  const DisjointRegions &dstRegions() const { return dstRegions_; }

  const chain::Chain &chain() const { return chain_; }

  bool operator==(const Path &p) const {
    return src() == p.src() && dst() == p.dst() && chain() == p.chain() &&
           dstRegions().equivalent(p.dstRegions());
  }

  bool operator!=(const Path &rhs) const { return !operator==(rhs); }

  std::string str() const;
  void append(std::ostream &) const;

private:
  TensorId src_;
  chain::Chain chain_;
  TensorId dst_;
  DisjointRegions dstRegions_;
};

using Paths = std::vector<Path>;

std::ostream &operator<<(std::ostream &, const Path &);
std::ostream &operator<<(std::ostream &, const Paths &);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
