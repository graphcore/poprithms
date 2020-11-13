// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_PROPOSAL_HPP
#define POPRITHMS_MEMORY_INPLACE_PROPOSAL_HPP
#include <algorithm>
#include <memory>
#include <vector>

#include <poprithms/memory/inplace/aliastype.hpp>
#include <poprithms/memory/inplace/tensorid.hpp>
#include <poprithms/memory/inplace/usings.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

/** A proposal for inplacing, consists of
 * 1) A TensorId, which is the output of the Op being proposed for inplacing
 * 2) An AliasType, which is the proposed inplace type.
 * */
class Proposal {
public:
  Proposal(TensorId id_, AliasType t_) : tensorId_(id_), type_(t_) {
    // Proposals cannot be to make an Op outplace, as it is not clear which
    // topological constraints can be removed in this case.
    // A proposal can also not be to make an Op be None AliasType.
    assertValidType();
  }

  TensorId tensorId() const { return tensorId_; }
  AliasType type() const { return type_; }

  void append(std::ostream &) const;

private:
  void assertValidType() const;

  TensorId tensorId_;
  AliasType type_;
};
using Proposals = std::vector<Proposal>;

std::ostream &operator<<(std::ostream &, const Proposal &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
