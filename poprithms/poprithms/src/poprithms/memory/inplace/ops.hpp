// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_OPS_HPP
#define POPRITHMS_MEMORY_INPLACE_OPS_HPP
#include "op.hpp"

#include <poprithms/memory/inplace/crossalias.hpp>

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

private:
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  alias::Color color_;
};

/** Concatenation */
class Concat : public Op {
public:
  Concat(const State &st, uint64_t axis__) : Op(st), axis_(axis__) {}
  uint64_t axis() const { return axis_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

private:
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  uint64_t axis_;
};

/** UnaryModifier (sqrt, etc) */
class UnaryModifier : public Op {
public:
  UnaryModifier(const State &st) : Op(st) {}
  std::string typeString() const final { return "UnaryModifier"; }
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return true; }

private:
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

/** Generalization of slice and subSample */
class SettSample : public Op {
public:
  SettSample(const State &st, const Region &region__)
      : Op(st), region_(region__) {}
  Region region() const { return region_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

private:
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  Region region_;
};

/** Multi-dimensional transpose */
class DimShuffle : public Op {
public:
  DimShuffle(const State &st, const Permutation &permutation__)
      : Op(st), permutation_(permutation__) {}
  Permutation permutation() const { return permutation_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;

  bool modifies(InIndex) const final { return false; }

private:
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  Permutation permutation_;
};

class Reverse : public Op {
public:
  Reverse(const State &st, const Dimensions &dimensions__)
      : Op(st), dimensions_(dimensions__) {}
  Dimensions dimensions() const { return dimensions_; }
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

private:
  bool typeSpecificEqualTo(const Op &other) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  Dimensions dimensions_;
};

class Reshape : public Op {
public:
  Reshape(const State &st) : Op(st) {}
  std::string typeString() const final { return "Reshape"; }
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

private:
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

class Identity : public Op {
public:
  Identity(const State &st) : Op(st) {}
  std::string typeString() const final { return "Identity"; }
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

private:
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
};

class Expand : public Op {
public:
  Expand(const State &st) : Op(st) {}
  std::string typeString() const final { return "Expand"; }
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final { return false; }

private:
  bool typeSpecificEqualTo(const Op &) const final { return true; }
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
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
class Multi : public Op {
public:
  Multi(const State &st, const CrossAliases &m__);
  std::string typeString() const final;
  std::unique_ptr<Op> clone() const final;
  bool modifies(InIndex) const final;
  const CrossAliases &mapping() const { return mapping_; }

private:
  bool typeSpecificEqualTo(const Op &) const final;
  AliasTensorIds typeSpecificGrow(alias::Graph &,
                                  const TensorMap &) const final;
  CrossAliases mapping_;
  std::vector<bool> inIndexIsModified_;
};

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
