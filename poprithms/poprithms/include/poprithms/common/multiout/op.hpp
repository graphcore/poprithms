// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_OP_HPP
#define POPRITHMS_COMMON_MULTIOUT_OP_HPP

#include <algorithm>
#include <ios>
#include <map>
#include <memory>
#include <sstream>
#include <typeinfo>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/contiguoussubset.hpp>

namespace poprithms {
namespace common {
namespace multiout {

class Graph;

using ContiguousOutIndexSubset = poprithms::util::ContiguousSubset<OutIndex>;
using ContiguousInIndexSubset  = poprithms::util::ContiguousSubset<InIndex>;
using Shape                    = ndarray::Shape;
using Shapes                   = ndarray::Shapes;

/**
 * Abstract base class of nodes in a multiout::Graph.
 *
 * A node in a multiout::Graph, which has multiple input and output Tensors,
 * each of which has a Shape. In addition to Shapes of the output Tensors,
 * each Op keeps track of which Ops consume its output Tensors.
 *
 * All inputs have an InIndex and all outputs have an OutIndex. These must be
 * contiguous, so if there is an input (output) at index i != 0, then there is
 * necessarily also an input (output) at index i-1.
 *
 * Ops store all attributes of their output Tensors. In this base class, the
 * data is just the 'Shape', but in derived classes it will be other things,
 * like type, device, etc. To be able to obtain the attributes of an Op's
 * inputs, Ops store a constant pointer to their containing Graph. See for
 * example how the method 'inShape' is implemented.
 *  */
class Op {

public:
  /** All Op member variables */
  struct State {

  public:
    State(const OpId id_,
          const TensorIds &inIds_,
          const std::vector<ConsumptionIds> &consumptionIds_,
          const Shapes &outShapes_,
          const std::string &name_,
          const Graph &mulitoutGraph_);

    // This Op's unique identifier.
    const OpId id;

    // The input Tensors of this Op, in order if InputIndex.
    const TensorIds inIds;

    // The Ops which consume the output Tensors of this Op, ordered by
    // OutIndex.
    const std::vector<ConsumptionIds> consumptionIds;

    // The Shapes of the output Tensors which this Op creates.
    const Shapes outShapes;

    // (optional) name to be associated to this Op, can be useful for logging.
    const std::string name;

    // The Graph which this Op belongs to.
    const Graph &multioutGraph;

    // The input Shapes are obtained from #inIds, by going via multioutGraph.
    Shapes inShapes() const;
    Shape inShape(InIndex) const;

    // Will be  "=default" in C++20, but for now must be done manually.
    bool operator==(const State &rhs) const;
    bool operator!=(const State &rhs) const { return !operator==(rhs); }

    uint64_t nIns() const { return inIds.size(); }
    uint64_t nOuts() const { return outShapes.size(); }
  };

  virtual ~Op();
  Op &operator=(const Op &) = default;
  Op &operator=(Op &&) = default;
  Op(const Op &)       = default;
  Op(Op &&)            = default;
  Op()                 = delete;

  Op(const State &ob);

  std::string str() const;

  OpId id() const { return id_; }

  /** The Shape if the #i'th input to this Op. */
  Shape inShape(InIndex i) const;

  /** The rank of the #i'th input to this Op. */
  uint64_t inRank(InIndex i) const;

  /** The number of elements in the #i'th input to this Op. */
  uint64_t nInElms(InIndex i) const;

  /** The Shape if the #i'th output of this Op. */
  const Shape &outShape(OutIndex o) const { return outShapes_.at(o.get()); }

  /** The rank of the #i'th output of this Op. */
  uint64_t outRank(OutIndex o) const { return outShape(o).rank_u64(); }

  /** The number of elements in the #i'th output of this Op. */
  uint64_t nOutElms(OutIndex o) const { return outShape(o).nelms_u64(); }

  /** The places where the Tensors created by this Op are consumed. */
  const std::vector<ConsumptionIds> &consumptionIds() const {
    return consumptionIds_;
  }

  /** The number of consumption ids of each output tensor.  */
  std::vector<uint64_t> nConsumptionIds() const;

  /** The total number of conumption ids, of all output tensors. */
  uint64_t totalConsumptionIds() const;

  uint64_t nConsumptionIds(OutIndex o) const {
    return consumptionIds(o).size();
  }

  bool hasConsumptionIds() const { return totalConsumptionIds() != 0; }

  bool hasConsumptionIds(OutIndex o) const { return nConsumptionIds(o) != 0; }

  /** The places where the #o'th Tensor created by this Op is consumed. */
  const ConsumptionIds &consumptionIds(OutIndex o) const {
    return consumptionIds_[o.get()];
  }

  /** return true if #c is a consumer of the output of this op at @o */
  bool isConsumptionId(OutIndex o, const ConsumptionId &c) const;

  /** The Shapes of the inputs of this Op, for each InIndex.  */
  Shapes inShapes() const;

  const Shapes &outShapes() const { return outShapes_; }

  const std::string &getName() const { return name_; }

  void setName(const std::string &n) { name_ = n; }

  State getState() const {
    return State{
        id_, inIds_, consumptionIds_, outShapes_, name_, *multioutGraph_};
  }

  /** The Tensors which this Op consumes. */
  const TensorIds &inTensorIds() const { return inIds_; }

  /** The #i'th Tensor which this Op consumes. */
  const TensorId &inTensorId(InIndex i) const { return inIds_[i.get()]; }

  /** The inputs at a subset of the input indices */
  TensorIds inTensorIds(const InIndices &) const;

  /** The inputs at all input indices except those in #exclude. */
  TensorIds inTensorIdsExcluding(const InIndices &exclude) const;

  uint64_t nInTensors() const { return inIds_.size(); }

  /**
   * The concatenation of the TensorIds of all input and output Tensors.
   * */
  TensorIds inAndOutTensorIds() const;

  /**
   * Ops have outputs at contiguous indices, which means optional outputs
   * are not supported in this Graph/Op.
   * */
  TensorIds outTensorIds() const;
  TensorId outTensorId(OutIndex o) const { return {id(), o}; }
  uint64_t nOutTensors() const { return outShapes().size(); }
  TensorIds outTensorIds(const OutIndices &) const;

  /**
   * The output indices of all the output Tensors which have at least one
   * consuming Op.
   * */
  std::vector<OutIndex> outIndicesConsumed() const;

  /**
   * All the InIndices of Op #id. These are [0, ..., nInTensors()).
   */
  InIndices inIndices() const;

  /**
   * All the OutIndices of Op #id. These are [0, ..., nOutTensors()).
   */
  OutIndices outIndices() const;

  /**
   * \sa multiOutTypeSpecificEqualTo. */
  bool operator==(const Op &rhs) const;

  /**
   * String describing the exact transformation performed by this Op
   * */
  virtual std::string typeString() const = 0;

  /**
   * Clone this Op, returning a unique_ptr to an indentical copy of it.
   * */
  virtual std::unique_ptr<Op> cloneMultioutOp() const = 0;

  std::unique_ptr<Op> clone() const { return cloneMultioutOp(); }

  /** Verify that the input and output indices are valid for this Op. If they
   * are not, a descriptive error message which includes #context is thrown.
   */
  void verify(InIndex, OutIndex, const std::string &context) const;

  /**
   * Verify that the input indices are all less than the total number of
   * inputs, and are distinct from each other.
   * */
  void verifyDistinct(const InIndices &indices) const;

  /**
   * Verify that the output indices are all less than the total number of
   * outputs, and are distinct from each other.
   * */
  void verifyDistinct(const OutIndices &indices) const;

  /**
   * Sometimes identical code patterns are used for input and output tensors.
   * The use of this enum class can reduce code duplication.
   * */
  enum class Port { In, Out };

  /**
   * returns "in" for Port::In and "out" for Port::Out.
   * */
  static std::string lowercase(Port);

  /**
   * The number of input/output tensors.
   * */
  uint64_t nTensors(Port) const;

  /**
   * The shape of the tensor at input/output index #i.
   * */
  Shape shape(Port, uint64_t i) const;

  /**
   * The id of the tensor at input/output index #i.
   * */
  TensorId tensorId(Port, uint64_t i) const;

private:
  OpId id_;
  TensorIds inIds_;
  std::vector<ConsumptionIds> consumptionIds_;
  Shapes outShapes_;
  std::string name_;
  const Graph *multioutGraph_;
  void setGraph(const Graph &);

protected:
  const Graph &multioutGraph() const { return *multioutGraph_; }

private:
  /**
   * A pure virtual function that derived classes must implement. This
   * function has a precondition that it will only be called when the 'other'
   * is the same type as the instance invoking the function.
   * */
  virtual bool multiOutTypeSpecificEqualTo(const Op &other) const = 0;

  void insertConsumptionId(OutIndex o, const ConsumptionId &c);

  /**
   * Remove #toRemove as a ConsumptionId of the output tensor at #o. If
   * #toRemove is not a ConsumptionId of this output tensor, an error is
   * thrown.
   * */
  void removeConsumptionId(OutIndex o, const ConsumptionId &toRemove);

  void resetInTensorId(InIndex i, const TensorId &id);

  friend class multiout::Graph;

protected:
  [[noreturn]] void unimplemented() const;
};

std::ostream &operator<<(std::ostream &, const Op &);

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
