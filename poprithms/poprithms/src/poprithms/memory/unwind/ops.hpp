// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_OPS_HPP
#define POPRITHMS_MEMORY_INPLACE_OPS_HPP

#include <memory/unwind/op.hpp>

#include <poprithms/memory/unwind/path.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using util::Permutation;
using UpMultioutOp = std::unique_ptr<poprithms::common::multiout::Op>;
using ndarray::Dimensions;
using ndarray::Shape;
using nest::Region;

class NonInput : public Op {
public:
  NonInput(const State &st) : Op(st) {}
  virtual bool isSink(OutIndex) const { return false; }
  virtual bool isSource(OutIndex) const { return false; }
};

class SumLike : public NonInput {
public:
  SumLike(const State &st, const std::vector<InIndex> &unwindIndices);
  std::string typeString() const final;
  void extendFwd(Chain &, InIndex, OutIndex) const final;
  void extendBwd(Chain &, InIndex, OutIndex) const final;
  UpMultioutOp cloneMultioutOp() const final { return Op::mu<SumLike>(this); }
  const std::vector<InIndex> &unwindIndices() const { return uwis; }
  bool isUnwindIndex(InIndex i) const {
    return std::find(uwis.cbegin(), uwis.cend(), i) != uwis.cend();
  }

  bool isUnwindable(InIndex i, OutIndex) const final {
    return isUnwindIndex(i);
  }
  bool isBarrier(OutIndex) const final { return false; }

private:
  std::vector<InIndex> uwis;
  bool unwindTypeSpecificEqualTo(const Op &) const final;
};

class Input : public Op {
public:
  Input(const State &st) : Op(st) {}
  void extendFwd(Chain &, InIndex, OutIndex) const final;
  void extendBwd(Chain &, InIndex, OutIndex) const final;
  bool isBarrier(OutIndex) const final { return false; }
};

class Sink : public Input {
public:
  Sink(const State &st) : Input(st) {}
  std::string typeString() const final { return "Sink"; }
  UpMultioutOp cloneMultioutOp() const final { return Op::mu<Sink>(this); }
  virtual bool isSink(OutIndex) const { return true; }
  virtual bool isSource(OutIndex) const { return false; }
  bool isUnwindable(InIndex, OutIndex) const final { return false; }

private:
  bool unwindTypeSpecificEqualTo(const Op &) const final { return true; }
};

/** Concatenation */
class Concat : public NonInput {
public:
  Concat(const State &st, uint64_t axis__)
      : NonInput(st), axis_(axis__),
        partitionPoints_(
            Shape::concatPartitionPoints(st.baseState.inShapes(), axis__)) {}
  uint64_t axis() const { return axis_; }
  std::string typeString() const final;
  void extendFwd(Chain &, InIndex, OutIndex) const final;
  void extendBwd(Chain &, InIndex, OutIndex) const final;
  UpMultioutOp cloneMultioutOp() const final { return Op::mu<Concat>(this); }
  bool isUnwindable(InIndex, OutIndex) const final { return true; }
  bool isBarrier(OutIndex) const final { return false; }

private:
  bool unwindTypeSpecificEqualTo(const Op &other) const final;
  uint64_t axis_;

  // the indices along the axis of concatenation where the concatenated
  // Tensors touch.
  const std::vector<int64_t> partitionPoints_;

  std::vector<int64_t> getLowerSlice(InIndex) const;
  std::vector<int64_t> getUpperSlice(InIndex) const;
};

/**
 * View-change operators with 1 input and 1 output.
 * */
class ViewChange1to1 : public NonInput {
public:
  ViewChange1to1(const Op::State &st);
  virtual bool isUnwindable(InIndex, OutIndex) const { return true; }
  void extendFwd(Chain &, InIndex, OutIndex) const final;
  void extendBwd(Chain &, InIndex, OutIndex) const final;
  bool isBarrier(OutIndex) const final { return false; }

private:
  virtual void fwd(Chain &) const = 0;
  virtual void bwd(Chain &) const = 0;
};

/** Generalization of slice and subSample */
class SettSample : public ViewChange1to1 {
public:
  SettSample(const State &st, const Region &region__)
      : ViewChange1to1(st), region_(region__) {}
  Region region() const { return region_; }
  std::string typeString() const final;
  UpMultioutOp cloneMultioutOp() const final {
    return Op::mu<SettSample>(this);
  }

private:
  void fwd(Chain &) const final;
  void bwd(Chain &) const final;
  bool unwindTypeSpecificEqualTo(const Op &other) const final;
  Region region_;
};

/** Multi-dimensional transpose */
class DimShuffle : public ViewChange1to1 {
public:
  DimShuffle(const State &st, const Permutation &permutation__)
      : ViewChange1to1(st), permutation_(permutation__) {}
  Permutation permutation() const { return permutation_; }
  std::string typeString() const final;
  UpMultioutOp cloneMultioutOp() const final {
    return Op::mu<DimShuffle>(this);
  }

private:
  void fwd(Chain &) const final;
  void bwd(Chain &) const final;
  bool unwindTypeSpecificEqualTo(const Op &other) const final;
  Permutation permutation_;
};

class Reverse : public ViewChange1to1 {
public:
  Reverse(const State &st, const Dimensions &dimensions__)
      : ViewChange1to1(st), dimensions_(dimensions__) {}
  Dimensions dimensions() const { return dimensions_; }
  std::string typeString() const final;
  UpMultioutOp cloneMultioutOp() const final { return Op::mu<Reverse>(this); }

private:
  void fwd(Chain &) const final;
  void bwd(Chain &) const final;
  bool unwindTypeSpecificEqualTo(const Op &other) const final;
  Dimensions dimensions_;
};

class Expand : public ViewChange1to1 {
public:
  Expand(const State &st);
  std::string typeString() const final { return "Expand"; }
  UpMultioutOp cloneMultioutOp() const final { return Op::mu<Expand>(this); }

private:
  void fwd(Chain &) const final;
  void bwd(Chain &) const final;
  bool unwindTypeSpecificEqualTo(const Op &) const final { return true; }
};

class Reshape : public ViewChange1to1 {
public:
  Reshape(const State &st);
  std::string typeString() const final { return "Reshape"; }
  UpMultioutOp cloneMultioutOp() const final { return Op::mu<Reshape>(this); }

private:
  void fwd(Chain &) const final;
  void bwd(Chain &) const final;
  bool unwindTypeSpecificEqualTo(const Op &) const final { return true; }
};

class Identity : public ViewChange1to1 {
public:
  Identity(const State &st) : ViewChange1to1(st) {}
  std::string typeString() const final { return "Identity"; }
  UpMultioutOp cloneMultioutOp() const final {
    return Op::mu<Identity>(this);
  }

private:
  void fwd(Chain &) const final {}
  void bwd(Chain &) const final {}
  bool unwindTypeSpecificEqualTo(const Op &) const final { return true; }
};

/**
 * An Op which takes multiple inputs and has multiple outputs, where the
 * semantics of how every output Tensor either modifies, aliases, or uses each
 * input Tensor is defined by a CrossLinks object.
 *
 * This Op can cover all use cases which do not involve non-trivial
 * view-changes (reshapes, dimShuffles, etc).
 **/
class BaseBarrier : public NonInput {
public:
  BaseBarrier(const State &st) : NonInput(st) {}
  virtual bool isUnwindable(InIndex, OutIndex) const { return false; }
  void extendFwd(Chain &, InIndex, OutIndex) const final;
  void extendBwd(Chain &, InIndex, OutIndex) const final;
  bool isBarrier(OutIndex) const final { return true; }
};

class Barrier : public BaseBarrier {
public:
  Barrier(const State &st) : BaseBarrier(st) {}
  std::string typeString() const final { return "Barrier"; }
  UpMultioutOp cloneMultioutOp() const final { return Op::mu<Barrier>(this); }

private:
  bool unwindTypeSpecificEqualTo(const Op &) const final { return true; }
};

class SumLikeReduce : public BaseBarrier {
public:
  SumLikeReduce(const State &st) : BaseBarrier(st) {}
  std::string typeString() const final { return "SumLikeReduce"; }
  UpMultioutOp cloneMultioutOp() const final {
    return Op::mu<SumLikeReduce>(this);
  }

private:
  bool unwindTypeSpecificEqualTo(const Op &) const final { return true; }
};

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
