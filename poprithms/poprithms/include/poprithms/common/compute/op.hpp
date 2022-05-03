// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_OP_HPP
#define POPRITHMS_COMMON_COMPUTE_OP_HPP

#include <poprithms/autodiff/core/togradgraph.hpp>
#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/program/callstack/callstack.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::ContiguousInIndexSubset;
using poprithms::common::multiout::ContiguousOutIndexSubset;
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::memory::nest::DisjointRegions;
using poprithms::memory::nest::Region;
using poprithms::ndarray::Dimensions;
using poprithms::ndarray::DType;
using poprithms::ndarray::DTypes;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;
using poprithms::program::callstack::CallEvent;
using poprithms::program::callstack::CallEvents;

class Graph;

using namespace poprithms::common;

using Lower = Shape::Lower;
using Upper = Shape::Upper;

using Shapes = std::vector<Shape>;
using DTypes = std::vector<DType>;

// An op in a graph
class Op : public schedulable::Op {

public:
  /** All Op member variables */
  struct State {

  public:
    State(const schedulable::Op::State &baseState_,
          const DTypes &outDTypes_,
          const std::vector<CallEvents> &inCopies_,
          const std::vector<CallEvents> &outCopies_,
          const Graph &pGraph_)
        : baseState(baseState_), outDTypes(outDTypes_), inCopies(inCopies_),
          outCopies(outCopies_), pGraph(&pGraph_) {}

    const schedulable::Op::State baseState;

    /**
     * The numerical type of each of the outputs of this op.
     * */
    const DTypes outDTypes;

    /**
     * All of the call events in the graph which involve a copy from a calling
     * sub-graph into an output tensor of this op. This happens when this
     * op's sub-graph in the callee.
     *
     * Specifically, inCopies[outIndex] is the call events where there is a
     * copy to into this op's outIndex'th output.
     * */
    const std::vector<CallEvents> inCopies;

    /**
     * All of the call events in the graph which involve a copy from this an
     * output of this op to the calling sub-graph. This happens when this op's
     * sub-graph in the callee.
     * */
    const std::vector<CallEvents> outCopies;

    Shape inShape(uint64_t i) const { return baseState.baseState.inShape(i); }

    /**
     * The graph to which this op belongs.
     * */
    const Graph *pGraph;

    bool operator==(const State &rhs) const;
    bool operator!=(const State &rhs) const { return !operator==(rhs); }
  };

  Op(const State &s);

  Op &operator=(const Op &) = default;
  virtual ~Op() override    = default;
  Op(const Op &)            = default;
  Op(Op &&)                 = default;
  Op()                      = delete;

  /**
   * Get the State of this compute::Op.
   * */
  State getComputeState() const;

  /**
   * This op does not store its input tensor types, so this call will go via
   * this op's graph -- ops only store their output types.
   * */
  DType inDType(InIndex i) const;

  /**
   * The output type of this op's #o'th output.
   * */
  DType outDType(OutIndex o) const { return outDTypes_.at(o.get()); }

  /**
   * The numerical type of the input/output (depending on #p) at index #i.
   * */
  DType dtype(Port p, uint64_t i) const;

  /**
   * The graph to which this op belongs.
   * */
  const Graph &graph() const;

public:
  void insertInCopy(OutIndex, const CallEvent &);
  void insertOutCopy(OutIndex, const CallEvent &);

  void removeInCopy(OutIndex, const CallEvent &);
  void removeOutCopy(OutIndex, const CallEvent &);

  /**
   * All call events which begin with a copy into the #o'th output tensor of
   * this op.
   * */
  const CallEvents &inCopies(OutIndex o) const {
    return inCopies_.at(o.get());
  }

  /**
   * All call events which begin with a copy into one of this op's output
   * tensors, from a tensor in the calling graph (this op is in a callee
   * graph).
   * */
  const std::vector<CallEvents> &inCopies() const { return inCopies_; }

  /**
   * All call events which end with a copy from the #o'th output tensor of
   * this op into the calling graph.
   * */
  const CallEvents &outCopies(OutIndex o) const {
    return outCopies_.at(o.get());
  }

  const std::vector<CallEvents> &outCopies() const { return outCopies_; }

  /**
   * Remove the inputs of this op at the indices defined by #insToRemove.
   * */
  void computeOpRemoveInputs(const ContiguousInIndexSubset &insToRemove);

  /**
   * Remove the outputs of this op at the indices defined by #insToRemove.
   * */
  void computeOpRemoveOutputs(const ContiguousOutIndexSubset &);

protected:
  // unique pointer to base op class.
  using UpBop = std::unique_ptr<multiout::Op>;

  /**
   * Some utility methods used for checking for correctness of ops where there
   * is an expectation on the equivalence of input/output types.
   * */
  void verifyInsSameDType() const;
  void verifyOutsSameDType() const;
  void verifyAllSameDType() const;

private:
  // See the comments in the Op::State class about these attributes.
  DTypes outDTypes_;
  std::vector<CallEvents> inCopies_;
  std::vector<CallEvents> outCopies_;
  const Graph *pGraph_;
};

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
