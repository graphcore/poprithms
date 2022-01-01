// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_SCHEDULABLE_OP_HPP
#define POPRITHMS_COMMON_SCHEDULABLE_OP_HPP

#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

using multiout::OpId;
using multiout::OpIds;
using multiout::TensorIds;

class Graph;

/**
 * A node in a Graph. It extends its base class, multiout::Op, by adding
 *
 * 1) input and output "control" dependencies, which needn't be data
 *    dependencies, and
 * 2) a sub-graph identifier.
 * */
class Op : public common::multiout::Op {

public:
  virtual ~Op() = default;
  Op &operator=(const Op &) = default;
  Op &operator=(Op &&) = default;
  Op(const Op &)       = default;
  Op(Op &&)            = default;
  Op()                 = delete;

  /** All Op member variables */
  struct State {

  public:
    State(const common::multiout::Op::State &baseState_,
          const SubGraphId subGraphId_,
          const OpIds &controlDependencyInOps_,
          const OpIds &controlDependencyOutOps_)
        : baseState(baseState_), subGraphId(subGraphId_),
          controlDependencyInOps(controlDependencyInOps_),
          controlDependencyOutOps(controlDependencyOutOps_) {}

    // The base state, contains Shapes, data dependencies (TensorIds), a name,
    // an OpId, etc.
    const common::multiout::Op::State baseState;

    const SubGraphId subGraphId;

    // Ops which must be scheduled before this Op, for non-data dependency
    // reasons.
    const OpIds controlDependencyInOps;

    // Ops which must be scheduled after this Op, for non-data dependency
    // reasons.
    const OpIds controlDependencyOutOps;

    // (will be  "=default" in C++20)
    bool operator==(const State &rhs) const;
  };

  Op(const State &ob);

  /**
   * \return the Op::State of this Op.
   * */
  State getSchedulableState() const;

  /**
   * Ops which must be scheduled before this Op, for non-data dependency
   * reasons.
   * */
  const OpIds &controlDependencyInOps() const {
    return controlDependencyInOps_;
  }

  /**
   * Ops which must be scheduled after this Op, for non-data dependency
   * reasons.
   * */
  const OpIds &controlDependencyOutOps() const {
    return controlDependencyOutOps_;
  }

  SubGraphId subGraphId() const { return subGraphId_; }

private:
  SubGraphId subGraphId_;

  // ALL control dependencies, including the data dependencies.
  OpIds controlDependencyInOps_;
  OpIds controlDependencyOutOps_;
  bool
  multiOutTypeSpecificEqualTo(const common::multiout::Op &other) const final;

  virtual bool schedulableTypeSpecificEqualTo(const Op &other) const = 0;

  /**
   * Insert an input control dependency, if it does not already exist.
   * */
  void insertControlDependencyIn(OpId);

  /** Remove an input control dependency, if it exists (else do nothing). */
  void removeControlDependencyIn(OpId);

  /**
   * Insert an output control dependency, if it does not already exist.
   * */
  void insertControlDependencyOut(OpId);

  /** Remove an output control dependency, if it exists (else do nothing). */
  void removeControlDependencyOut(OpId);

  // only the schedulable::Graph can modify an op after it's constructed, no
  // class which inherits from it.
  friend class schedulable::Graph;
};

} // namespace schedulable
} // namespace common
} // namespace poprithms

#endif
