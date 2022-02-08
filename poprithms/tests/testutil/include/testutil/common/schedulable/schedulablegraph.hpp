// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_COMMON_SCHEDULABLE_SCHEDULABLEGRAPH_HPP
#define TESTUTIL_COMMON_SCHEDULABLE_SCHEDULABLEGRAPH_HPP

#include <iostream>
#include <vector>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>

namespace poprithms {
namespace common {
namespace schedulable_test {

using namespace poprithms::common;
using Shape  = poprithms::ndarray::Shape;
using Shapes = poprithms::ndarray::Shapes;
using multiout::ContiguousInIndexSubset;
using multiout::ContiguousOutIndexSubset;
using multiout::InIndices;
using multiout::OpId;
using multiout::OpIds;
using multiout::OptionalTensorIds;
using multiout::TensorId;
using multiout::TensorIds;
using schedulable::SubGraphId;

class Op : public schedulable::Op {
public:
  Op(const schedulable::Op::State &s);
  std::string typeString() const final;
  std::unique_ptr<multiout::Op> cloneMultioutOp() const final;
  void
  removeSchedulableDerivedOutputs(const ContiguousOutIndexSubset &) final {
    // nothing to do: no new attributes.
  }

  void removeSchedulableDerivedInputs(const ContiguousInIndexSubset &) final {
    // nothing to do: no new attributes.
  }

private:
  bool schedulableTypeSpecificEqualTo(const schedulable::Op &) const final {
    // nothing to do: no new attributes.

    return true;
  }
};

class Graph : public schedulable::Graph {
public:
  using schedulable::Graph::removeOp;

  OpId insert(const TensorIds &ins,
              uint64_t nOut,
              SubGraphId sgId,
              std::string name);

  virtual ~Graph() override;
  void appendOpColumns(std::ostream &, const OpIds &) const final;

  OpId insertBinBoundary(schedulable::SubGraphId sgId) final;

private:
  bool multiOutTypeSpecificEqualTo(const multiout::Graph &) const final {
    // nothing to do: no new attributes.
    return true;
  }

  void schedulableTypeSpecificRemoveOp(OpId,
                                       const OptionalTensorIds &) final {
    // nothing to do.
  }

  void schedulableTypeSpecificVerifyValidOutputSubstitute(
      const TensorId &,
      const TensorId &) const final {
    // nothing to do.
  }
};

std::ostream &operator<<(std::ostream &, const Graph &);

} // namespace schedulable_test
} // namespace common
} // namespace poprithms

#endif
