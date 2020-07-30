// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_OP_HPP
#define POPRITHMS_MEMORY_ALIAS_OP_HPP

#include <map>
#include <memory>
#include <sstream>

#include <poprithms/memory/alias/origins.hpp>
#include <poprithms/memory/alias/tensor.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/shape.hpp>

namespace poprithms {
namespace memory {
namespace alias {

/**
 * A Node in a Graph. This class represents relationships between
 * Tensors, and properties of Tensors: how they are combined, their sizes, how
 * they alias each other, etc.
 * */
class Node {

public:
  /** All Node member variables */
  struct State {
    using Ids    = std::vector<TensorId>;
    using Shapes = std::vector<Shape>;

  public:
    State(const Ids &ins_,
          const Ids &outs_,
          const Shapes &inShapes_,
          TensorId id_,
          const Shape &sh_,
          const Origins &origins_)
        : ins(ins_), outs(outs_), inShapes(inShapes_), id(id_), shape(sh_),
          origins(origins_){};

    const Ids ins;
    const Ids outs;
    const Shapes inShapes;
    const TensorId id;
    const Shape shape;
    const Origins origins;
  };

  virtual ~Node() = default;

  Node(const State &ob)
      : ins_(ob.ins), outs_(ob.outs), inShapes_(ob.inShapes), id_(ob.id),
        shape_(ob.shape), origins_(ob.origins) {}

  /** Clone this Node with a potentially different State. Attributes in the
   * derived classes are cloned exactly.
   * */
  virtual std::unique_ptr<Node> clone(const State &) const = 0;

  /** An exact clone of this Node */
  std::unique_ptr<Node> clone() const { return clone(getState()); }

  /** String describing the exact transformation */
  virtual std::string typeString() const = 0;

  /**  \return true iff this Node might alias a strict subset of its inputs.
   *           That is, not all inputs are aliased.
   */
  virtual bool samples() const = 0;

  /** \return true iff this Node has no inputs and creates a new Allocation */
  virtual bool allocates() const = 0;

  /** Map output regions to input regions */
  virtual DisjointRegions
  getInRegions(InIndex, const DisjointRegions &thisRegions) const = 0;

  std::string str() const { return typeString() + std::string("::") + id(); }

  TensorId in(InIndex i) const { return ins_[i.get()]; }

  TensorId id() const { return id_; }

  const std::vector<TensorId> &ins() const { return ins_; }

  const std::vector<TensorId> &outs() const { return outs_; }

  /** \return ins() and outs() concatenated */
  std::vector<TensorId> insAndOuts() const;

  int nIns_i32() const { return static_cast<int>(ins().size()); }

  State getState() const {
    return State{ins_, outs_, inShapes_, id_, shape_, origins_};
  }

  const Shape &shape() const { return shape_; }

  const Shape &inShape(uint64_t i) const { return inShapes_[i]; }

  const std::vector<Shape> &inShapes() const { return inShapes_; }

  void insertOut(TensorId id);

  bool containsAliases() const { return origins_.containsAliases(); }

  bool isAliasedTo(const Node &rhs) const {
    return origins_.isAliasedTo(rhs.origins_);
  }

  void clearOrigins() const { origins_.clear(); }

  void insertOrigin(AllocId id, const DisjointRegions &r) const {
    origins_.insert(id, r);
  }

  void insertOriginsFrom(const Node &rhs) const {
    origins_.insert(rhs.origins_);
  }

  bool isRowMajorSetContiguous() const {
    return origins_.isRowMajorSetContiguous();
  }

  std::vector<AllocId> getAllocIds() const { return origins_.getAllocIds(); }

  const Origins &origins() const { return origins_; }

  bool operator==(const Node &rhs) const;

private:
  const std::vector<TensorId> ins_;
  std::vector<TensorId> outs_;
  const std::vector<Shape> inShapes_;
  const TensorId id_;
  const Shape shape_;

  // mutable, acts like a cache and is completely hidden from user. Making it
  // mutable makes for a cleaner API.
  mutable Origins origins_;
};

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
