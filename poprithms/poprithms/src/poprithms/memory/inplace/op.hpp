// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_OP_HPP
#define POPRITHMS_MEMORY_INPLACE_OP_HPP
#include <algorithm>
#include <ios>
#include <map>
#include <memory>
#include <sstream>

#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/memory/inplace/consumer.hpp>
#include <poprithms/memory/inplace/proposal.hpp>
#include <poprithms/memory/inplace/tensorid.hpp>
#include <poprithms/memory/inplace/tensormap.hpp>
#include <poprithms/memory/inplace/usings.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

/** A node in an inplace::Graph, with directed edges (control dependencies /
 * topological constraints) to and from other Ops, input and output Tensors,
 * a name, and an AliasType. Inheriting classes can be found in ops.hpp.
 *  */
class Op {

public:
  /** All Op member variables */
  struct State {

  public:
    State(const OpId id_,
          const OpIds &ins_,
          const OpIds &outs_,
          const TensorIds &inIds_,
          const std::vector<Consumers> &consumers_,
          const Shapes &outShapes_,
          const std::string &name_)
        : id(id_), ins(ins_), outs(outs_), inIds(inIds_),
          consumers(consumers_), outShapes(outShapes_), name(name_) {}

    // This Op's unique identifier
    const OpId id;

    // Dependencies that this Op has. In other words, Ops which must be
    // scheduled before this Op
    const OpIds ins;

    // Ops which have dependencies on this Op. In other words, Ops which must
    // be scheduled after this Op
    const OpIds outs;

    // The input Tensors of this Op, in order if InputIndex
    const TensorIds inIds;

    // The Ops which consume output Tensors of this Op, in the order of
    // OutputIndex
    const std::vector<Consumers> consumers;

    // The Shapes of the output Tensors which this Op creates
    const Shapes outShapes;

    // (optional) name to be associated to this Op, can be useful for logging
    const std::string name;

    // Will be  "=default" in C++20, but for now must be done manually.
    bool operator==(const State &rhs) const;
  };

  virtual ~Op();
  Op &operator=(const Op &) = default;
  Op &operator=(Op &&) = default;
  Op(const Op &)       = default;
  Op(Op &&)            = default;
  Op()                 = delete;

  Op(const State &ob);

  /** Ops which must be scheduled before this Op. */
  const OpIds &ins() const { return ins_; }

  /** Ops which must be scheduled after this Op. */
  const OpIds &outs() const { return outs_; }

  std::string str() const { return typeString() + std::string("::") + id(); }

  OpId id() const { return id_; }

  const Shape &outShape(OutIndex i) const { return outShapes_.at(i.get()); }

  uint64_t outRank(OutIndex i) const { return outShape(i).rank_u64(); }

  uint64_t nOutElms(OutIndex i) const { return outShape(i).nelms_u64(); }

  const std::vector<Consumers> &consumers() const { return consumers_; }

  const Consumers &consumers(OutIndex o) const { return consumers_[o.get()]; }

  const Shapes &outShapes() const { return outShapes_; }

  const std::string &name() const { return name_; }

  void setName(const std::string &n) { name_ = n; }

  State getState() const {
    return State{id_, ins_, outs_, inIds_, consumers_, outShapes_, name_};
  }

  const TensorIds &inTensorIds() const { return inIds_; }
  const TensorId &inTensorId(InIndex i) const { return inIds_[i.get()]; }
  uint64_t nInTensors() const { return inIds_.size(); }

  /** Note that Ops must have outputs at contiguous indices, which means
   * optional outputs are not supported in this Graph/Op project.
   * */
  TensorIds outTensorIds() const;
  TensorId outTensorId(OutIndex o) const { return {id(), o}; }
  uint64_t nOutTensors() const { return outShapes().size(); }

  void insertIn(OpId);
  void insertOut(OpId);
  void insertConsumer(OutIndex, const Consumer &);

  bool operator==(const Op &rhs) const {
    return getState() == rhs.getState() && typeid(*this) == typeid(rhs) &&
           typeSpecificEqualTo(rhs);
  }

  /**
   * String describing the exact transformation performed by this Op
   * */
  virtual std::string typeString() const = 0;

  /**
   * Append this Op's alias::Graph equivalent(s) into \a g, and also
   * insert the mapping between this Op's input and output Tensors and the
   * alias::Graph's equivalents into \a m.
   *
   * \param g. This is the Graph which contains full information about how
   *           Tensors are composed of allocations, and how they alias each
   *           other. The alias::Graph class is similar to poplar::Graph.
   *
   * \param m A mapping between TensorIds in this Ops Graph, and TensorIds in
   *          \a g.
   * */
  void grow(alias::Graph &g, TensorMap &m) const;

  /** \return true iff (if and only if) the input at InIndex \a i is modified
   */
  virtual bool modifies(InIndex i) const = 0;

  /** \return true iff the input is modified at any InIndex */
  bool modifies() const;

  /** \return all InIndex where the input is modified */
  std::vector<InIndex> modifyingIndices() const;

  virtual std::unique_ptr<Op> clone() const = 0;

  using AliasTensorIds = std::vector<alias::TensorId>;
  using OutIndices     = std::vector<OutIndex>;
  using InIndices      = std::vector<InIndex>;

  static State getBaseState(const OpId opId,
                            const TensorIds &tensorIns,
                            const Shapes &outShapes,
                            const OpIds &opIns);

private:
  OpId id_;
  OpIds ins_;
  OpIds outs_;
  TensorIds inIds_;
  std::vector<Consumers> consumers_;
  Shapes outShapes_;
  std::string name_;

private:
  /**
   * A pure virtual function that derived classes must implement.
   * This function has a precondition that it will only
   * be called when the 'other' is the same type as the instance
   * invoking the function.
   * */
  virtual bool typeSpecificEqualTo(const Op &other) const = 0;

  virtual AliasTensorIds typeSpecificGrow(alias::Graph &,
                                          const TensorMap &) const = 0;
};

std::ostream &operator<<(std::ostream &, const Op &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
