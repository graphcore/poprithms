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

namespace poprithms {
namespace common {
namespace multiout {

using Shape  = ndarray::Shape;
using Shapes = ndarray::Shapes;

/**
 * Abstract base class of nodes in a multiout::Graph.
 *
 * A node in a multiout::Graph, which has multiple input and output Tensors,
 * each of which has a Shape. In addition to the input and output Shapes, each
 * Op keeps track of which Ops consume its output Tensors.
 *
 * All inputs have an InIndex and all outputs have an OutIndex. These must be
 * contiguous, so if there is an input (output) at index i != 0, then there is
 * necessarily also an input (output) at index i-1.
 *  */
class Op {

public:
  /** All Op member variables */
  struct State {

  public:
    State(const OpId id_,
          const TensorIds &inIds_,
          const std::vector<ConsumptionIds> &consumptionIds_,
          const Shapes &inShapes_,
          const Shapes &outShapes_,
          const std::string &name_);

    // This Op's unique identifier
    const OpId id;

    // The input Tensors of this Op, in order if InputIndex
    const TensorIds inIds;

    // The Ops which consume the output Tensors of this Op, ordered by
    // OutIndex/
    const std::vector<ConsumptionIds> consumptionIds;

    // The Shapes of the input Tensors of this Op
    const Shapes inShapes;

    // The Shapes of the output Tensors which this Op creates
    const Shapes outShapes;

    // (optional) name to be associated to this Op, can be useful for logging
    const std::string name;

    // Will be  "=default" in C++20, but for now must be done manually.
    bool operator==(const State &rhs) const;
    bool operator!=(const State &rhs) const { return !operator==(rhs); }

    uint64_t nIns() const { return inShapes.size(); }
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
  const Shape &inShape(InIndex i) const { return inShapes_.at(i.get()); }

  /** The rank of the #i'th input to this Op. */
  uint64_t inRank(InIndex i) const { return inShape(i).rank_u64(); }

  /** The number of elements in the #i'th input to this Op. */
  uint64_t nInElms(InIndex i) const { return inShape(i).nelms_u64(); }

  /** The Shape if the #i'th output of this Op. */
  const Shape &outShape(OutIndex i) const { return outShapes_.at(i.get()); }

  /** The rank of the #i'th output of this Op. */
  uint64_t outRank(OutIndex i) const { return outShape(i).rank_u64(); }

  /** The number of elements in the #i'th output of this Op. */
  uint64_t nOutElms(OutIndex i) const { return outShape(i).nelms_u64(); }

  /** The places where the Tensors created by this Op are consumed. */
  const std::vector<ConsumptionIds> &consumptionIds() const {
    return consumptionIds_;
  }

  /** The places where the #o'th Tensor created by this Op is consumed. */
  const ConsumptionIds &consumptionIds(OutIndex o) const {
    return consumptionIds_[o.get()];
  }

  const Shapes &inShapes() const { return inShapes_; }

  const Shapes &outShapes() const { return outShapes_; }

  const std::string &getName() const { return name_; }

  void setName(const std::string &n) { name_ = n; }

  State getState() const {
    return State{id_, inIds_, consumptionIds_, inShapes_, outShapes_, name_};
  }

  /** The Tensors which this Op consumes. */
  const TensorIds &inTensorIds() const { return inIds_; }

  /** The #i'th Tensor which this Op consumes. */
  const TensorId &inTensorId(InIndex i) const { return inIds_[i.get()]; }

  uint64_t nInTensors() const { return inIds_.size(); }

  /**
   * Ops must have outputs at contiguous indices, which means optional outputs
   * are not supported in this Graph/Op.
   * */
  TensorIds outTensorIds() const;
  TensorId outTensorId(OutIndex o) const { return {id(), o}; }
  uint64_t nOutTensors() const { return outShapes().size(); }

  void insertConsumptionId(OutIndex o, const ConsumptionId &c) {
    consumptionIds_[o.get()].push_back(c);
  }

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
  virtual std::unique_ptr<Op> clone() const = 0;

  /** Verify that the input and output indices are valid for this Op. If they
   * are not, a descriptive error message which includes #context is thrown.
   */
  void verify(InIndex, OutIndex, const std::string &context) const;

private:
  OpId id_;
  TensorIds inIds_;
  std::vector<ConsumptionIds> consumptionIds_;
  Shapes inShapes_;
  Shapes outShapes_;
  std::string name_;

private:
  /**
   * A pure virtual function that derived classes must implement. This
   * function has a precondition that it will only be called when the 'other'
   * is the same type as the instance invoking the function.
   * */
  virtual bool multiOutTypeSpecificEqualTo(const Op &other) const = 0;
};

std::ostream &operator<<(std::ostream &, const Op &);

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
