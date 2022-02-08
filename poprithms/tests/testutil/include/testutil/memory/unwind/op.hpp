// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_TESTUTIL_UNWIND_TOY_OP_HPP
#define POPRITHMS_TESTUTIL_UNWIND_TOY_OP_HPP

#include <poprithms/common/multiout/op.hpp>
#include <poprithms/common/schedulable/graph.hpp>
#include <poprithms/common/schedulable/op.hpp>
#include <poprithms/memory/unwind/matmulattractions.hpp>
#include <poprithms/memory/unwind/scheduledsolution.hpp>

namespace poprithms {
namespace unwindtoy {

class FullState;

using MatMulAttractions = poprithms::memory::unwind::MatMulAttractions;
using common::multiout::ConsumptionId;
using common::multiout::ConsumptionIds;
using common::multiout::InIndex;
using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::OutIndex;
using common::multiout::TensorId;
using common::multiout::TensorIds;
using memory::unwind::ScheduledSolution;
using ndarray::Shape;
using ndarray::Shapes;
using util::Permutation;
using Lower         = ndarray::Shape::Lower;
using MultioutOp    = common::multiout::Op;
using Upper         = ndarray::Shape::Upper;
using HTensor       = compute::host::Tensor;
using HTensors      = std::vector<HTensor>;
using SchedulableOp = common::schedulable::Op;
using memory::unwind::Path;
using State = SchedulableOp::State;
using common::multiout::ContiguousInIndexSubset;
using common::multiout::ContiguousOutIndexSubset;

HTensor getMatMulOut(const Shape &s0, const Shape &s1);

class Op : public SchedulableOp {
public:
  Op(const State &st) : SchedulableOp(st) {}

  // This class is just for testing, so we're not going to support Graph or Op
  // comparison.
  bool schedulableTypeSpecificEqualTo(const SchedulableOp &) const final {
    unimplemented();
  }

  void growUnwind(FullState &u) const;

  void
  removeSchedulableDerivedOutputs(const ContiguousOutIndexSubset &) final {
    unimplemented();
  }

  void removeSchedulableDerivedInputs(const ContiguousInIndexSubset &) final {
    unimplemented();
  }

  // Create the host Tensors of the output of this op.
  virtual void fwd(FullState &) const = 0;

private:
  // Append to the unwind::Graph of fs.
  virtual TensorIds grow(FullState &fs) const = 0;
};

class MatMul : public Op {
public:
  MatMulAttractions atts_;
  MatMul(const State &st, const MatMulAttractions &atts)
      : Op(st), atts_(atts) {}
  std::unique_ptr<MultioutOp> cloneMultioutOp() const final;
  void fwd(FullState &fs) const final;
  std::string typeString() const final { return "MatMul"; }

private:
  TensorIds grow(FullState &u) const final;
};

class Reduce : public Op {
public:
  Reduce(const State &st) : Op(st) {}
  std::unique_ptr<MultioutOp> cloneMultioutOp() const final;
  void fwd(FullState &fs) const final;
  std::string typeString() const final { return "Reduce"; }

private:
  TensorIds grow(FullState &u) const final;
};

class Slice : public Op {
public:
  Lower lower_;
  Upper upper_;
  Slice(const State &st, const Lower &l, const Upper &u)
      : Op(st), lower_(l), upper_(u) {}
  std::unique_ptr<MultioutOp> cloneMultioutOp() const final;
  void fwd(FullState &fs) const final;
  std::string typeString() const final;

private:
  TensorIds grow(FullState &u) const final;
};

class Sum : public Op {
public:
  std::vector<InIndex> unwindables_;
  memory::unwind::SumAttractions sassy;
  Sum(const State &st,
      const std::vector<InIndex> &us,
      const memory::unwind::SumAttractions &sassy_)
      : Op(st), unwindables_(us), sassy(sassy_) {}
  std::unique_ptr<MultioutOp> cloneMultioutOp() const final;
  void fwd(FullState &fs) const final;
  std::string typeString() const final { return "Sum"; }

private:
  TensorIds grow(FullState &u) const final;
};

class DimShuffle : public Op {
public:
  Permutation p_;
  DimShuffle(const State &st, const Permutation &p) : Op(st), p_(p) {}
  std::unique_ptr<MultioutOp> cloneMultioutOp() const final;
  void fwd(FullState &fs) const final;
  std::string typeString() const final;

private:
  TensorIds grow(FullState &u) const final;
};

class Expand : public Op {
public:
  Expand(const State &st) : Op(st) {}
  std::unique_ptr<MultioutOp> cloneMultioutOp() const final;
  void fwd(FullState &fs) const final;
  std::string typeString() const final;

private:
  TensorIds grow(FullState &u) const final;
};

class Input : public Op {
public:
  double linear_;
  Input(const State &st, double l) : Op(st), linear_(l) {}
  std::unique_ptr<MultioutOp> cloneMultioutOp() const final;
  void fwd(FullState &fs) const final;
  std::string typeString() const final { return "Input"; }

private:
  TensorIds grow(FullState &u) const final;
};

class Concat : public Op {
public:
  uint64_t axis_;
  Concat(const State &st, uint64_t a) : Op(st), axis_(a) {}
  std::unique_ptr<MultioutOp> cloneMultioutOp() const final;
  void fwd(FullState &fs) const final;
  std::string typeString() const final;

private:
  TensorIds grow(FullState &u) const final;
};

} // namespace unwindtoy
} // namespace poprithms
#endif
