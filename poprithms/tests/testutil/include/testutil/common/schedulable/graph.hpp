// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_COMMON_SCHEDULABLE_GRAPH_HPP
#define TESTUTIL_COMMON_SCHEDULABLE_GRAPH_HPP

#include <vector>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>

namespace poprithms {
namespace common {

/**
 * A minimal completion of the abstract schedulable::Graph class.
 * */
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
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;
using schedulable::SubGraphId;

class Op final : public schedulable::Op {
public:
  Op(const schedulable::Op::State &s, bool phobic);
  std::string typeString() const final;
  std::unique_ptr<multiout::Op> cloneMultioutOp() const final;

  bool isConstraintPhobic() const final { return phobic_; }

private:
  bool
  schedulableTypeSpecificEqualTo(const schedulable::Op &rhs) const final {
    return phobic_ == rhs.isConstraintPhobic();
  }

private:
  bool phobic_;
};

class Graph final : public schedulable::Graph {
public:
  using schedulable::Graph::removeOp;

  void verifySchedulableDerivedGraphValid() const final {}

  OpId insert(const TensorIds &ins,
              uint64_t nOut,
              SubGraphId sgId,
              const std::string &name,
              bool isPhobic = false);

  OpId insertPhobic(const TensorIds &ins,
                    uint64_t nOut,
                    SubGraphId sgId,
                    const std::string &name) {
    return insert(ins, nOut, sgId, name, true);
  }

  std::map<OpId, OpIds>
  schedulableDerivedSpecificConstraints(const OpIds &) const final {
    return {};
  }

  void
  multiOutTypeSpecificRemoveInputs(OpId,
                                   const ContiguousInIndexSubset &) final {}

  void multiOutTypeSpecificRemoveOutputs(OpId,
                                         const ContiguousOutIndexSubset &,
                                         const OptionalTensorIds &) final {}

  virtual ~Graph() override;
  void appendOpColumns(std::ostream &, const OpIds &) const final;

  OpId insertBinBoundary(schedulable::SubGraphId sgId) final;

private:
  bool multiOutTypeSpecificEqualTo(const multiout::Graph &) const final {
    // no new attributes (nothing == nothing).
    return true;
  }

  void schedulableTypeSpecificRemoveOp(OpId,
                                       const OptionalTensorIds &) final {
    // nothing to do: no new attributes.
  }

  void
  schedulableTypeSpecificVerifyValidSubstitute(const TensorId &,
                                               const TensorId &) const final {
    // nothing to do: no new attributes.
  }
};

std::ostream &operator<<(std::ostream &, const Graph &);

} // namespace schedulable_test
} // namespace common
} // namespace poprithms

#endif
