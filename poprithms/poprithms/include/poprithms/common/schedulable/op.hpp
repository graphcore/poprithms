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
 * A node in a Graph. It extends its base class by addings
 *
 * - input and output dependencies, which needn't be data dependencies.
 * - a sub-graph identifier.
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
          const OpIds &inOps_,
          const OpIds &outOps_)
        : baseState(baseState_), subGraphId(subGraphId_), inOps(inOps_),
          outOps(outOps_) {}

    // The base state, contains Shapes, data dependencies (TensorIds), a name,
    // and OpId, etc.
    const common::multiout::Op::State baseState;

    const SubGraphId subGraphId;

    // Dependencies that this Op has. In other words, Ops which must be
    // scheduled before this Op. This includes all data dependencies.
    const OpIds inOps;

    // Ops which have dependencies on this Op. In other words, Ops which must
    // be scheduled after this Op. This includes all data dependencies.
    const OpIds outOps;

    // (will be  "=default" in C++20)
    bool operator==(const State &rhs) const;
  };

  Op(const State &ob);

  /**
   * Ops which must be scheduled before this Op. This includes all data
   * dependencies.
   * */
  const OpIds &inOps() const { return inOps_; }

  /**
   * Ops which must be scheduled after this Op. This includes all data
   * dependencies.
   * */
  const OpIds &outOps() const { return outOps_; }

  SubGraphId subGraphId() const { return subGraphId_; }

  /**
   * Return all of the input depencies which do not produce a tensor which
   * this Op consumes.
   * */
  OpIds nonDataInOps() const;

  /**
   * Return all of the output dependencies which do not consume an output
   * tensor of this Op.
   * */
  OpIds nonDataOutOps() const;

  bool isIn(OpId) const;

  bool isOut(OpId) const;

private:
  SubGraphId subGraphId_;

  // ALL control dependencies, including the data dependencies.
  OpIds inOps_;
  OpIds outOps_;
  bool
  multiOutTypeSpecificEqualTo(const common::multiout::Op &other) const final;

  virtual bool schedulableTypeSpecificEqualTo(const Op &other) const = 0;

  /**
   * Insert an input dependency, if it does not already exist.
   * */
  void insertIn(OpId);

  /** Remove an input dependency, if it exists (else do nothing). */
  void removeIn(OpId);

  /**
   * Insert an output dependency, if it does not already exist.
   * */
  void insertOut(OpId);

  /** Remove an output dependency, if it exists (else do nothing). */
  void removeOut(OpId);

  // only the schedulable::Graph can modify an op after it's constructed, no
  // class which inherits from it.
  friend class schedulable::Graph;
};

} // namespace schedulable
} // namespace common
} // namespace poprithms

#endif
