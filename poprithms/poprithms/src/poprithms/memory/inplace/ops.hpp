// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_OPS_HPP
#define POPRITHMS_MEMORY_INPLACE_OPS_HPP
#include "op.hpp"

#include <poprithms/memory/inplace/crosslink.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

class Mux : public Op {
public:
  /** An open Mux, flowing from input at index i_, to output. */
  Mux(const State &st, InIndex i_);

  /** A closed Mux */
  Mux(const State &st);

  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

  bool closed() const { return inIndex_ < 0; }
  bool open() const { return !closed(); }
  bool outplace() const { return closed(); }
  InIndex inIndex() const;

  void openAt(alias::Graph &g, TensorMap &m, InIndex);
  void close(alias::Graph &, TensorMap &);

  DisjointRegions
  outRegions(const DisjointRegions &in, InIndex, OutIndex) const final;
  DisjointRegions
  inRegions(const DisjointRegions &, InIndex, OutIndex) const final;

private:
  int64_t inIndex_{-1};
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

/** Allocations, with no inputs, 1 output (either constant or variable) */
class Alloc : public Op {
public:
  Alloc(const State &st, alias::Color color__) : Op(st), color_(color__) {}
  alias::Color color() const { return color_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

  DisjointRegions
  outRegions(const DisjointRegions &in, InIndex, OutIndex) const final;
  DisjointRegions
  inRegions(const DisjointRegions &, InIndex, OutIndex) const final;

private:
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  alias::Color color_;
};

/** Concatenation */
class Concat : public Op {
public:
  Concat(const State &st, uint64_t axis__)
      : Op(st), axis_(axis__),
        partitionPoints_(Shape::concatPartitionPoints(st.inShapes, axis__)) {}
  uint64_t axis() const { return axis_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

  DisjointRegions
  outRegions(const DisjointRegions &in, InIndex, OutIndex) const final;
  DisjointRegions
  inRegions(const DisjointRegions &, InIndex, OutIndex) const final;

private:
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  uint64_t axis_;

  // the indices along the axis of concatenation where the concatenated
  // Tensors touch.
  const std::vector<int64_t> partitionPoints_;

  std::vector<int64_t> getLowerSlice(InIndex) const;
  std::vector<int64_t> getUpperSlice(InIndex) const;
};

/** UnaryModifier (sqrt, etc) */
class UnaryModifier : public Op {
public:
  UnaryModifier(const State &st) : Op(st) {}
  std::string typeString() const final { return "UnaryModifier"; }
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return true; }

  DisjointRegions
  outRegions(const DisjointRegions &in, InIndex, OutIndex) const final {
    return in;
  }
  DisjointRegions
  inRegions(const DisjointRegions &out, InIndex, OutIndex) const final {
    return out;
  }

private:
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

class ViewChange1to1 : public Op {
public:
  ViewChange1to1(const State &st) : Op(st) {}
  bool modifies(InIndex) const final { return false; }

  DisjointRegions
  outRegions(const DisjointRegions &in, InIndex, OutIndex) const final;
  DisjointRegions
  inRegions(const DisjointRegions &, InIndex, OutIndex) const final;

private:
  virtual DisjointRegions outRegs(const DisjointRegions &in) const = 0;
  virtual DisjointRegions inRegs(const DisjointRegions &) const    = 0;
};

/** Generalization of slice and subSample */
class SettSample : public ViewChange1to1 {
public:
  SettSample(const State &st, const Region &region__)
      : ViewChange1to1(st), region_(region__) {}
  Region region() const { return region_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;

private:
  DisjointRegions outRegs(const DisjointRegions &in) const final {
    return in.settSample(region());
  }
  DisjointRegions inRegs(const DisjointRegions &) const final;
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  Region region_;
};

/** Multi-dimensional transpose */
class DimShuffle : public ViewChange1to1 {
public:
  DimShuffle(const State &st, const Permutation &permutation__)
      : ViewChange1to1(st), permutation_(permutation__) {}
  Permutation permutation() const { return permutation_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;

private:
  DisjointRegions outRegs(const DisjointRegions &in) const final;
  DisjointRegions inRegs(const DisjointRegions &) const final;
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  Permutation permutation_;
};

class Reverse : public ViewChange1to1 {
public:
  Reverse(const State &st, const Dimensions &dimensions__)
      : ViewChange1to1(st), dimensions_(dimensions__) {}
  Dimensions dimensions() const { return dimensions_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;

private:
  DisjointRegions outRegs(const DisjointRegions &in) const final {
    return in.reverse(dimensions().get());
  }

  DisjointRegions inRegs(const DisjointRegions &) const final;
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  Dimensions dimensions_;
};

class Reshape : public ViewChange1to1 {
public:
  Reshape(const State &st);
  std::string typeString() const final { return "Reshape"; }
  std::unique_ptr<Op> clone() const final;

private:
  DisjointRegions outRegs(const DisjointRegions &in) const final {
    return in.reshape(outShape(0));
  }
  DisjointRegions inRegs(const DisjointRegions &) const final;
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

class Identity : public ViewChange1to1 {
public:
  Identity(const State &st) : ViewChange1to1(st) {}
  std::string typeString() const final { return "Identity"; }
  std::unique_ptr<Op> clone() const final;

private:
  DisjointRegions outRegs(const DisjointRegions &in) const final {
    return in;
  }
  DisjointRegions inRegs(const DisjointRegions &out) const final {
    return out;
  }
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

class Expand : public ViewChange1to1 {
public:
  Expand(const State &st) : ViewChange1to1(st) {}
  std::string typeString() const final { return "Expand"; }
  std::unique_ptr<Op> clone() const final;

private:
  DisjointRegions outRegs(const DisjointRegions &in) const final;
  DisjointRegions inRegs(const DisjointRegions &) const final;
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

/**
 * An Op which takes multiple inputs and has multiple outputs, where the
 * semantics of how every output Tensor either modifies, aliases, or uses each
 * input Tensor is defined by a CrossLinks object.
 *
 * This Op can cover all use cases which do not involve non-trivial
 * view-changes (reshapes, dimShuffles, etc).
 **/
class Multi : public Op {
public:
  Multi(const State &st, const CrossLinks &m__);
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final;

  DisjointRegions
  outRegions(const DisjointRegions &in, InIndex, OutIndex) const final;
  DisjointRegions
  inRegions(const DisjointRegions &, InIndex, OutIndex) const final;

private:
  bool typeSpecificEqualTo(const Op &) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  CrossLinks mapping_;
  std::vector<bool> inIndexIsModified_;
  const CrossLinks &mapping() const { return mapping_; }
};

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
