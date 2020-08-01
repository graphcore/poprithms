// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_GBASE_NODE_HPP
#define POPRITHMS_MEMORY_GBASE_NODE_HPP

#include <map>
#include <memory>
#include <sstream>

#include <poprithms/memory/gbase/gbaseusings.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/shape.hpp>

namespace poprithms {
namespace memory {
namespace gbase {

/**
 * A Node in a Graph which represents a basic Tensor type, and it's
 * relationship to the Tensors which it derives from (inputs) and the Tensors
 * which derive from it (outputs).
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
          const Shape &sh_)
        : ins(ins_), outs(outs_), inShapes(inShapes_), id(id_), shape(sh_) {}

    const Ids ins;
    const Ids outs;
    const Shapes inShapes;
    const TensorId id;
    const Shape shape;
  };

  virtual ~Node() = default;

  Node(const State &ob)
      : ins_(ob.ins), outs_(ob.outs), inShapes_(ob.inShapes), id_(ob.id),
        shape_(ob.shape) {}

  /** String describing the exact transformation */
  virtual std::string typeString() const = 0;

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
    return State{ins_, outs_, inShapes_, id_, shape_};
  }

  const Shape &shape() const { return shape_; }

  const Shape &inShape(uint64_t i) const { return inShapes_[i]; }

  const std::vector<Shape> &inShapes() const { return inShapes_; }

  void insertOut(TensorId id);

  bool operator==(const Node &rhs) const;

private:
  const std::vector<TensorId> ins_;
  std::vector<TensorId> outs_;
  const std::vector<Shape> inShapes_;
  const TensorId id_;
  const Shape shape_;
};

} // namespace gbase
} // namespace memory
} // namespace poprithms

#endif
