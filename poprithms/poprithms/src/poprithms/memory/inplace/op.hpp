// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_OP_HPP
#define POPRITHMS_MEMORY_INPLACE_OP_HPP

#include <ios>
#include <memory>
#include <sstream>
#include <typeinfo>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/memory/inplace/tensormap.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

using common::multiout::ConsumptionId;
using common::multiout::ConsumptionIds;
using common::multiout::InIndex;
using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::OutIndex;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using ndarray::Shape;
using ndarray::Shapes;
using poprithms::common::multiout::ContiguousInIndexSubset;
using poprithms::common::multiout::ContiguousOutIndexSubset;
class Graph;

/** A node in an inplace::Graph, with directed edges (control dependencies /
 * topological constraints) to and from other Ops, input and output Tensors,
 * a name, and an AliasType. Inheriting classes can be found in ops.hpp.
 *  */
class Op : public common::multiout::Op {

public:
  /** All Op member variables */
  struct State {

  public:
    State(const common::multiout::Op::State &state,
          const OpIds &ins_,
          const OpIds &outs_)
        : baseState(state), ins(ins_), outs(outs_) {}

    State(const OpId id_,
          const TensorIds &inIds_,
          const std::vector<ConsumptionIds> &consumptionIds_,
          const Shapes &outShapes_,
          const std::string &name_,
          const OpIds &ins_,
          const OpIds &outs_,
          const Graph &g_);

    const common::multiout::Op::State baseState;

    // Dependencies that this Op has. In other words, Ops which must be
    // scheduled before this Op
    const OpIds ins;

    // Ops which have dependencies on this Op. In other words, Ops which must
    // be scheduled after this Op
    const OpIds outs;

    // Will be  "=default" in C++20, but for now must be done manually.
    bool operator==(const State &rhs) const;
  };

  virtual ~Op();
  Op &operator=(const Op &) = default;
  Op &operator=(Op &&) = default;
  Op(const Op &)       = default;
  Op(Op &&)            = default;
  Op()                 = delete;

  Op(const State &ob)
      : common::multiout::Op(ob.baseState), ins_(ob.ins), outs_(ob.outs) {}

  /** Ops which must be scheduled before this Op. */
  const OpIds &ins() const { return ins_; }
  uint64_t nIns() const { return ins_.size(); }

  /** Ops which must be scheduled after this Op. */
  const OpIds &outs() const { return outs_; }
  uint64_t nOuts() const { return outs_.size(); }

  State getState() const {
    return State{common::multiout::Op::getState(), ins_, outs_};
  }

  void insertOut(OpId);
  void insertIn(OpId);

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

  static State getStartingState(const OpId opId,
                                const TensorIds &tensorIns,
                                const Shapes &outShapes,
                                const Graph &);

  using AliasTensorIds = std::vector<alias::TensorId>;

  /**
   * Is the value of the output at #o a view of the input at #i? This does not
   * include inplace modifiers.
   * */
  virtual bool isView(InIndex i, OutIndex o) const = 0;

  bool isViewOfAnyOutput(InIndex i) const;

private:
  OpIds ins_;
  OpIds outs_;

private:
  /**
   * A pure virtual function that derived classes must implement.
   * This function has a precondition that it will only
   * be called when the 'other' is the same type as the instance
   * invoking the function.
   * */
  virtual bool inplaceTypeSpecificEqualTo(const Op &other) const = 0;

  virtual AliasTensorIds typeSpecificGrow(alias::Graph &,
                                          const TensorMap &) const = 0;

  bool
  multiOutTypeSpecificEqualTo(const common::multiout::Op &other) const final;

  void removeMultioutDerivedOutputs(const ContiguousOutIndexSubset &) final {
    unimplemented();
  }

  void removeMultioutDerivedInputs(const ContiguousInIndexSubset &) final {
    unimplemented();
  }
};

std::ostream &operator<<(std::ostream &, const Op &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
