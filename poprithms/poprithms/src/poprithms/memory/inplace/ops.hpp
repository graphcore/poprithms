// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_OPS_HPP
#define POPRITHMS_MEMORY_INPLACE_OPS_HPP
#include "op.hpp"

#include <poprithms/memory/inplace/crossalias.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

/** Allocations, with no inputs, 1 output (either constant or variable) */
class Alloc : public Op {
public:
  Alloc(const State &st, alias::Color color__) : Op(st), color_(color__) {}
  alias::Color color() const { return color_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

private:
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final;
  void applyOutplaceTo(alias::Graph &, const TensorMap &) const final;
  bool typeSpecificEqualTo(const Op &other) const final;
  OutIndices outAliasIndicesIf(AliasType) const final { return {}; }
  InIndices inAliasIndicesIf(AliasType) const final { return {}; }
  InIndices inModifiedIndicesIf(AliasType) const final { return {}; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  alias::Color color_;
};

/** Abstract base class for Ops which are not Allocs, and have inputs */
class NonAlloc : public Op {
public:
  NonAlloc(const State &st) : Op(st) {}

private:
  void applyOutplaceTo(alias::Graph &, const TensorMap &) const final;
  virtual AliasTensorIds growInplace(alias::Graph &,
                                     const TensorMap &) const = 0;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

/** Concatenation */
class Concat : public NonAlloc {
public:
  Concat(const State &st, uint64_t axis__) : NonAlloc(st), axis_(axis__) {}
  uint64_t axis() const { return axis_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

private:
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final;
  bool typeSpecificEqualTo(const Op &other) const final;
  OutIndices outAliasIndicesIf(AliasType) const final;
  InIndices inAliasIndicesIf(AliasType) const final;
  InIndices inModifiedIndicesIf(AliasType) const final { return {}; }
  AliasTensorIds growInplace(alias::Graph &, const TensorMap &) const final;
  uint64_t axis_;
};

/** Unary (sqrt, etc) */
class Unary : public NonAlloc {
public:
  Unary(const State &st) : NonAlloc(st) {}
  std::string typeString() const final { return "Unary"; }
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final;

private:
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final;
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  OutIndices outAliasIndicesIf(AliasType) const final;
  InIndices inAliasIndicesIf(AliasType) const final;
  AliasTensorIds growInplace(alias::Graph &, const TensorMap &) const final;
  InIndices inModifiedIndicesIf(AliasType t) const final {
    return inAliasIndicesIf(t);
  }
};

/** Binary (add, sub) */
class Binary : public NonAlloc {
public:
  Binary(const State &st);
  std::string typeString() const final { return "Binary"; }
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final;

private:
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final;
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  OutIndices outAliasIndicesIf(AliasType) const final;
  InIndices inAliasIndicesIf(AliasType) const final;
  AliasTensorIds growInplace(alias::Graph &, const TensorMap &) const final;
  InIndices inModifiedIndicesIf(AliasType t) const final {
    return inAliasIndicesIf(t);
  }
  void assertShapesValid();
};

/** Virtual base for non-modififying "view" Ops with 1 input and 1 output */
class UnaryView : public NonAlloc {
public:
  UnaryView(const State &st) : NonAlloc(st) {}
  bool modifies(InIndex) const final { return false; }

private:
  OutIndices outAliasIndicesIf(AliasType) const final;
  InIndices inAliasIndicesIf(AliasType) const final;
  InIndices inModifiedIndicesIf(AliasType) const final { return {}; }
};

/** Generalization of slice and subSample */
class SettSample : public UnaryView {
public:
  SettSample(const State &st, const Region &region__)
      : UnaryView(st), region_(region__) {}
  Region region() const { return region_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;

private:
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final;
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds growInplace(alias::Graph &, const TensorMap &) const final;
  Region region_;
};

/** Multi-dimensional transpose */
class DimShuffle : public UnaryView {
public:
  DimShuffle(const State &st, const Permutation &permutation__)
      : UnaryView(st), permutation_(permutation__) {}
  Permutation permutation() const { return permutation_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;

private:
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds growInplace(alias::Graph &, const TensorMap &) const final;
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final;
  Permutation permutation_;
};

class Reverse : public UnaryView {
public:
  Reverse(const State &st, const Dimensions &dimensions__)
      : UnaryView(st), dimensions_(dimensions__) {}
  Dimensions dimensions() const { return dimensions_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;

private:
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds growInplace(alias::Graph &, const TensorMap &) const final;
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final;
  Dimensions dimensions_;
};

class Reshape : public UnaryView {
public:
  Reshape(const State &st) : UnaryView(st) {}
  std::string typeString() const final { return "Reshape"; }
  std::unique_ptr<Op> clone() const final;

private:
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final;
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds growInplace(alias::Graph &, const TensorMap &) const final;
};

class Identity : public UnaryView {
public:
  Identity(const State &st) : UnaryView(st) {}
  std::string typeString() const final { return "Identity"; }
  std::unique_ptr<Op> clone() const final;

private:
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final;
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds growInplace(alias::Graph &, const TensorMap &) const final;
};

class Expand : public UnaryView {
public:
  Expand(const State &st) : UnaryView(st) {}
  std::string typeString() const final { return "Expand"; }
  std::unique_ptr<Op> clone() const final;

private:
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final;
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds growInplace(alias::Graph &, const TensorMap &) const final;
};

/** Base class for Ops which only every have 1 AliasType, and will not be
 * considered for modification in, for example, the inplacing transformation
 */
class NoneAliasType : public Op {
public:
  NoneAliasType(const State &st);

private:
  void
  applyInplaceTo(alias::Graph &, const TensorMap &, AliasType) const final {
    invalidCall("applyInplaceTo");
  }
  void applyOutplaceTo(alias::Graph &, const TensorMap &) const final {
    invalidCall("applyOutplaceTo");
  }
  OutIndices outAliasIndicesIf(AliasType) const final {
    invalidCall("outAliasIndicesIf");
  }
  InIndices inAliasIndicesIf(AliasType) const final {
    invalidCall("inAliasIndicesIf");
  }
  InIndices inModifiedIndicesIf(AliasType) const final {
    invalidCall("inModifiedIndicesIf");
  }

  [[noreturn]] void invalidCall(const std::string &) const;
};

/**
 * Multi-input, multi-output Op, where any input can be aliased to any
 * (single) output, and can optionally be modified. This Op can cover all use
 * cases which do not involve non-trivial view-changes (reshapes, dimShuffles,
 * etc).
 *
 * An Op which does not have any aliasing between inputs and outputs will have
 * the mapping_ vector empty.
 **/
class Multi : public NoneAliasType {
public:
  using Mapping = std::vector<CrossAlias>;
  Multi(const State &st, const Mapping &m__);
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final;
  const Mapping &mapping() const { return mapping_; }

private:
  bool typeSpecificEqualTo(const Op &) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  Mapping mapping_;
  std::vector<bool> inIndexIsModified_;
};

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
