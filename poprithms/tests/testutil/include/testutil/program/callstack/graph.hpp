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
class Op : public schedulable::Op {

public:
  Op(const schedulable::Op::State &s,
     const SubGraphIds &callees,
     const CopyIns &inCopies,
     const CopyOuts &outCopies)
      : schedulable::Op(s), callees_(callees), inCopies_(inCopies),
        outCopies_(outCopies) {}

  std::string typeString() const final;

  std::unique_ptr<multiout::Op> cloneMultioutOp() const final;

  /**
   * Remove the copy-outs at specified output indices (and remove indices,
   * shifting other output indices down as necessary to keep contiguous
   * indices). */
  void removeSchedulableDerivedOutputs(const ContiguousOutIndexSubset &);

  /**
   * Remove the copy-ins at specified input indices (and remove indices,
   * shifting other input indices down as necessary to keep contiguous
   * indices). */
  void removeSchedulableDerivedInputs(const ContiguousInIndexSubset &);

  const CopyIns &inCopies() const { return inCopies_; }
  const CopyOuts &outCopies() const { return outCopies_; }
  const SubGraphIds &callees() const { return callees_; }

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
};

/**
 * A minimal completion of the abstract schedulable::Graph class which allows
 * ops to have callees, and copies into and out of the callees.
 * */
class Graph : public schedulable::Graph {

private:
  schedulable::Op::State getState(const TensorIds &ins,
                                  uint64_t nOut,
                                  SubGraphId,
                                  const std::string &name) const;

public:
  using schedulable::Graph::removeOp;

  const Op &op(OpId id) const {
    return static_cast<const Op &>(multioutOp(id));
  }

  // insert a "normal" op which has no callees.
  OpId insert(const TensorIds &ins,
              uint64_t nOut,
              SubGraphId sgId,
              const std::string &name);

  // insert an op which has callees.
  OpId insert(SubGraphId sgId,
              const SubGraphIds &callees,
              const CopyIns &inCopies,
              const CopyOuts &outCopies,
              const std::string &name);

  const SubGraphIds &callees(OpId id) const { return op(id).callees(); }

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

  void schedulableTypeSpecificVerifyValidOutputSubstitute(
      const TensorId &,
      const TensorId &) const final {
    // nothing to do: no new attributes.
  }
};

std::ostream &operator<<(std::ostream &, const Graph &);

} // namespace callstack_test
} // namespace program
} // namespace poprithms

#endif