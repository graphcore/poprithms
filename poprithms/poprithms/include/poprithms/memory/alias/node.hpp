// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_OP_HPP
#define POPRITHMS_MEMORY_ALIAS_OP_HPP

#include <map>
#include <memory>
#include <sstream>

#include <poprithms/memory/alias/origins.hpp>
#include <poprithms/memory/alias/tensor.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace alias {

/**
 * A Node in a Graph which represents a basic Tensor with a Shape, and its
 * relationship to the Tensors which it derives from (inputs) and the Tensors
 * which derive from it (outputs).
 * */
class Node {

  /** All Node member variables */
  struct State {
    using Ids    = std::vector<TensorId>;
    using Shapes = std::vector<Shape>;

  public:
    State(const Ids &ins_,
          const Ids &outs_,
          const Shapes &inShapes_,
          TensorId id_,
          const Shape &sh_)
        : ins(ins_), outs(outs_), inShapes(inShapes_), id(id_), shape(sh_) {}

    const Ids ins;
    const Ids outs;
    const Shapes inShapes;
    const TensorId id;
    const Shape shape;
  };

public:
  using State = Node::State;

  virtual ~Node() = default;

  Node(const State &ob, const Origins &oris)
      : ins_(ob.ins), outs_(ob.outs), inShapes_(ob.inShapes), id_(ob.id),
        shape_(ob.shape), origins_(oris) {}

  /** String describing the exact transformation */
  virtual std::string typeString() const = 0;

  std::string str() const { return typeString() + std::string("::") + id(); }

  TensorId in(InIndex i) const { return ins_[i.get()]; }

  TensorId id() const { return id_; }

  const std::vector<TensorId> &ins() const { return ins_; }

  const std::vector<TensorId> &outs() const { return outs_; }

  /** \return ins() and outs() concatenated */
  std::vector<TensorId> insAndOuts() const;

  int nIns_i32() const { return static_cast<int>(ins().size()); }

  State getState() const {
    return State{ins_, outs_, inShapes_, id_, shape_};
  }

  const Shape &shape() const { return shape_; }

  const Shape &inShape(uint64_t i) const { return inShapes_[i]; }

  const std::vector<Shape> &inShapes() const { return inShapes_; }

  void insertOut(TensorId id);

  void removeOut(TensorId id);

  bool operator==(const Node &rhs) const;

  /** Clone this Node with a potentially different State. Attributes in the
   * derived classes are cloned exactly.
   * */
  virtual std::unique_ptr<Node> clone(const State &,
                                      const Origins &) const = 0;

  /** An exact clone of this Node */
  std::unique_ptr<Node> clone() const { return clone(getState(), origins()); }

  /**  \return true iff this Node might alias a strict subset of its inputs.
   *           That is, not all inputs are aliased.
   */
  virtual bool samples() const = 0;

  /** \return true iff this Node has no inputs and creates a new Allocation */
  virtual bool allocates() const = 0;

  /** Map output regions to input regions */
  virtual DisjointRegions
  getInRegions(InIndex, const DisjointRegions &thisRegions) const = 0;

  bool containsAliases() const { return origins_.containsAliases(); }

  bool isAliasedTo(const Node &rhs) const {
    return origins_.isAliasedTo(rhs.origins_);
  }

  void clearOrigins() { origins_.clear(); }

  void insertOrigin(AllocId id, const DisjointRegions &r) {
    origins_.insert(id, r);
  }

  void insertOriginsFrom(const Node &rhs) { origins_.insert(rhs.origins_); }

  bool isRowMajorSetContiguous() const {
    return origins_.isRowMajorSetContiguous();
  }

  std::vector<AllocId> getAllocIds() const { return origins_.getAllocIds(); }

  const Origins &origins() const { return origins_; }

private:
  const std::vector<TensorId> ins_;
  std::vector<TensorId> outs_;
  const std::vector<Shape> inShapes_;
  const TensorId id_;
  const Shape shape_;
  Origins origins_;
};

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
