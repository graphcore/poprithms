// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_OPS_HPP
#define POPRITHMS_MEMORY_INPLACE_OPS_HPP

#include <memory/inplace/op.hpp>
#include <poprithms/memory/inplace/crosslink.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

using ndarray::Dimensions;
using nest::Region;
using util::Permutation;

using UpMultioutOp = std::unique_ptr<poprithms::common::multiout::Op>;

class AliasGate : public Op {
public:
  /** An open AliasGate, flowing from input at index i_, to output. */
  AliasGate(const State &st, InIndex i_);

  /** A closed AliasGate */
  AliasGate(const State &st);

  std::string typeString() const final;
  UpMultioutOp cloneMultioutOp() const final;
  bool modifies(InIndex) const final { return false; }

  bool closed() const { return inIndex_ < 0; }
  bool open() const { return !closed(); }
  InIndex inIndex() const;

  void openAt(alias::Graph &g, TensorMap &m, InIndex);
  void close(alias::Graph &, TensorMap &);

private:
  int64_t inIndex_{-1};
  bool inplaceTypeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

/** Allocations, with no inputs, 1 output (either constant or variable) */
class Alloc : public Op {
public:
  Alloc(const State &st, alias::Color color__) : Op(st), color_(color__) {}
  alias::Color color() const { return color_; }
  std::string typeString() const final;
  UpMultioutOp cloneMultioutOp() const final;
  bool modifies(InIndex) const final { return false; }

private:
  bool inplaceTypeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  alias::Color color_;
};

/** Concatenation */
class Concat : public Op {
public:
  Concat(const State &st, uint64_t axis__)
      : Op(st), axis_(axis__),
        partitionPoints_(
            Shape::concatPartitionPoints(st.baseState.inShapes, axis__)) {}
  uint64_t axis() const { return axis_; }
  std::string typeString() const final;
  UpMultioutOp cloneMultioutOp() const final;
  bool modifies(InIndex) const final { return false; }

private:
  bool inplaceTypeSpecificEqualTo(const Op &other) const final;
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
class Unary : public Op {
public:
  Unary(const State &st) : Op(st) {}
};

/** UnaryModifier (sqrt, etc) */
class UnaryModifier : public Unary {
public:
  UnaryModifier(const State &st) : Unary(st) {}
  std::string typeString() const final { return "UnaryModifier"; }
  UpMultioutOp cloneMultioutOp() const final;
  bool modifies(InIndex) const final { return true; }

private:
  bool inplaceTypeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

class ViewChange1to1 : public Op {
public:
  ViewChange1to1(const State &st) : Op(st) {}
  bool modifies(InIndex) const final { return false; }

private:
};

/** Generalization of slice and subSample */
class SettSample : public ViewChange1to1 {
public:
  SettSample(const State &st, const Region &region__)
      : ViewChange1to1(st), region_(region__) {}
  Region region() const { return region_; }
  std::string typeString() const final;
  UpMultioutOp cloneMultioutOp() const final;

private:
  bool inplaceTypeSpecificEqualTo(const Op &other) const final;
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
  UpMultioutOp cloneMultioutOp() const final;

private:
  bool inplaceTypeSpecificEqualTo(const Op &other) const final;
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
  UpMultioutOp cloneMultioutOp() const final;

private:
  bool inplaceTypeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  Dimensions dimensions_;
};

class Reshape : public ViewChange1to1 {
public:
  Reshape(const State &st);
  std::string typeString() const final { return "Reshape"; }
  UpMultioutOp cloneMultioutOp() const final;

private:
  bool inplaceTypeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

class Identity : public ViewChange1to1 {
public:
  Identity(const State &st) : ViewChange1to1(st) {}
  std::string typeString() const final { return "Identity"; }
  UpMultioutOp cloneMultioutOp() const final;

private:
  bool inplaceTypeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

class Expand : public ViewChange1to1 {
public:
  Expand(const State &st) : ViewChange1to1(st) {}
  std::string typeString() const final { return "Expand"; }
  UpMultioutOp cloneMultioutOp() const final;

private:
  bool inplaceTypeSpecificEqualTo(const Op &) const final { return true; }
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
  UpMultioutOp cloneMultioutOp() const final;
  bool modifies(InIndex) const final;

private:
  bool inplaceTypeSpecificEqualTo(const Op &) const final;
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
