// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_PROPOSAL_HPP
#define POPRITHMS_MEMORY_INPLACE_PROPOSAL_HPP
#include <memory>
#include <vector>

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

using common::multiout::InIndex;
using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::OutIndex;
using common::multiout::TensorIds;

class Tensor;

class Proposal;
using Proposals = std::vector<Proposal>;

/** A proposal to open a Mux at a specific InIndex */
class Proposal {
public:
  Proposal(OpId id_, InIndex index) : muxId_(id_), index_(index) {}
  Proposal(const Tensor &, InIndex index);
  static Proposals open0(const OpIds &);
  static Proposals open0(const TensorIds &);
  InIndex inIndex() const { return index_; }
  OpId muxId() const { return muxId_; }
  void append(std::ostream &) const;

private:
  OpId muxId_;
  InIndex index_;
};

std::ostream &operator<<(std::ostream &, const Proposal &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
