// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef TESTUTIL_PROGRAM_CALLSTACK_GRAPH_AND_OP_HPP
#define TESTUTIL_PROGRAM_CALLSTACK_GRAPH_AND_OP_HPP

#include <sstream>
#include <vector>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/program/callstack/copyin.hpp>
#include <poprithms/program/callstack/copymap.hpp>
#include <poprithms/program/callstack/copyout.hpp>
#include <poprithms/program/callstack/querier.hpp>

namespace poprithms {
namespace program {
namespace callstack_test {

using poprithms::common::multiout::ConsumptionId;
using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::ContiguousInIndexSubset;
using poprithms::common::multiout::ContiguousOutIndexSubset;
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OptionalTensorId;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;
using poprithms::program::callstack::CalleeIndex;
using poprithms::program::callstack::CalleeIndices;
using poprithms::program::callstack::CallEvent;
using poprithms::program::callstack::CallEvents;
using poprithms::program::callstack::CallStack;
using poprithms::program::callstack::CopyIn;
using poprithms::program::callstack::CopyInMap;
using poprithms::program::callstack::CopyIns;
using poprithms::program::callstack::CopyOutMap;
using poprithms::program::callstack::CopyOuts;

using namespace poprithms::common;

/**
 * A minimal Op class for testing callstack functionality. Adds callees, input
 * copies, and output copies to the abstract class schedulable::Op.
 * */
class Op final : public schedulable::Op {

  friend class Graph;

public:
  Op(const schedulable::Op::State &s,
     const SubGraphIds &callees,
     const CopyIns &inCopies,
     const CopyOuts &outCopies,
     const std::vector<std::pair<TensorId, TensorId>> &carries)
      : schedulable::Op(s), callees_(callees), inCopies_(inCopies),
        outCopies_(outCopies), carries_(carries) {}

  bool isConstraintPhobic() const final { return false; }

  std::string typeString() const final;

  std::unique_ptr<multiout::Op> cloneMultioutOp() const final;

  const CopyIns &inCopies() const { return inCopies_; }

  const CopyOuts &outCopies() const { return outCopies_; }

  const SubGraphIds &callees() const { return callees_; }

  InIndices nonCalleeCopyInIndices() const;

  std::vector<std::pair<InIndex, TensorId>> copyInDsts() const;

  bool isCarriedTo(const TensorId &) const;

  TensorId carriedFrom(const TensorId &) const;

private:
  bool
  schedulableTypeSpecificEqualTo(const schedulable::Op &rhs) const final {
    // static cast is fine as this method is only called after the type of
    // #rhs has been established.
    auto cOp = static_cast<const Op *>(&rhs);
    return inCopies_ == cOp->inCopies() && outCopies_ == cOp->outCopies() &&
           callees_ == cOp->callees();
  }

  SubGraphIds callees_;
  CopyIns inCopies_;
  CopyOuts outCopies_;
  std::vector<std::pair<TensorId, TensorId>> carries_;
};

/**
 * A minimal completion of the abstract schedulable::Graph class which allows
 * ops to have callees, and copies into and out of the callees.
 * */
class Graph final : public schedulable::Graph {

  void verifySchedulableDerivedGraphValid() const final {}

private:
  schedulable::Op::State getState(const TensorIds &ins,
                                  uint64_t nOut,
                                  SubGraphId,
                                  const std::string &name) const;

  Op &mutableOp(OpId id) { return static_cast<Op &>(multioutOp(id)); }

public:
  using schedulable::Graph::removeOp;

  const Op &op(OpId id) const {
    return static_cast<const Op &>(multioutOp(id));
  }

  std::map<OpId, OpIds>
  schedulableDerivedSpecificConstraints(const OpIds &) const final {
    return {};
  }

  // insert a "normal" op. That is, an op which has no callees.
  OpId insert(const TensorIds &ins,
              uint64_t nOut,
              SubGraphId sgId,
              const std::string &name);

  // insert a generalized op with callees. The op has an optional input which
  // is not a copy (#condition) to model a switch op, and optional set of
  // tensors which are carried, to model a repeat op.
  //
  // carries[i] = {carriedTo, carriedFrom}.
  //
  OpId insert(SubGraphId sgId,
              const SubGraphIds &callees,
              const CopyIns &inCopies,
              const CopyOuts &outCopies,
              OptionalTensorId condition,
              const std::vector<std::pair<TensorId, TensorId>> &carries,
              const std::string &name);

  const SubGraphIds &callees(OpId id) const { return op(id).callees(); }

  virtual ~Graph() override;

  void appendOpColumns(std::ostream &, const OpIds &) const final;

  OpId insertBinBoundary(schedulable::SubGraphId sgId) final;

  bool isCarriedTo(const TensorId &tId, const CallStack &cs) const;

  TensorId carriedFrom(const TensorId &tId, const CallStack &cs) const;

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

  void
  multiOutTypeSpecificRemoveInputs(OpId,
                                   const ContiguousInIndexSubset &) final;

  void multiOutTypeSpecificRemoveOutputs(OpId,
                                         const ContiguousOutIndexSubset &,
                                         const OptionalTensorIds &) final;
};

std::ostream &operator<<(std::ostream &, const Graph &);

} // namespace callstack_test
} // namespace program
} // namespace poprithms

#endif
