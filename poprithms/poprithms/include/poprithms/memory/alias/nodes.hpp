// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_OPS_HPP
#define POPRITHMS_MEMORY_ALIAS_OPS_HPP

#include <poprithms/memory/alias/node.hpp>
namespace poprithms {
namespace memory {
namespace alias {

class Concat : public Node {
public:
  Concat(const State &ob, uint64_t a)
      : Node(ob), axis_(a),
        partitionPoints_(Shape::concatPartitionPoints(ob.inShapes, axis_)) {}

  /** \return The axis of concatenation */
  uint64_t axis() const { return axis_; }
  std::string typeString() const final { return "Concat"; }
  std::unique_ptr<Node> clone(const State &) const final;
  DisjointRegions getInRegions(InIndex, const DisjointRegions &) const final;

  /** \return false as all inputs are aliased */
  bool samples() const final { return false; }

  /** \return false as the output is just a view into the inputs, there are no
   *          new allocations/variables. */
  bool allocates() const final { return false; }

private:
  const uint64_t axis_;

  // the indices along the axis of concatenation where the concatenated
  // Tensors touch.
  const std::vector<int64_t> partitionPoints_;

  std::vector<int64_t> getLowerSlice(InIndex) const;
  std::vector<int64_t> getUpperSlice(InIndex) const;
};

class SettSample : public Node {
public:
  SettSample(const State &, const Shape &in_, const Lower &, const Upper &);
  SettSample(const State &ob, const Region &r) : Node(ob), region_(r){};

  /** \return The Region to sample the input Tensor at */
  const Region &region() const { return region_; }
  std::string typeString() const final;
  std::unique_ptr<Node> clone(const State &) const final;
  DisjointRegions getInRegions(InIndex, const DisjointRegions &) const final;
  bool samples() const final { return true; }
  bool allocates() const final { return false; }

private:
  const Region region_;
};

class Allocate : public Node {
public:
  Allocate(const State &ob, Color color) : Node(ob), color_(color) {}
  std::string typeString() const final { return "Allocate"; }
  std::unique_ptr<Node> clone(const State &) const final;
  DisjointRegions getInRegions(InIndex, const DisjointRegions &) const final;
  bool samples() const final { return false; }
  bool allocates() const final { return true; }
  Color color() const { return color_; }

private:
  Color color_;
};

class Reshape : public Node {
public:
  Reshape(const State &ob) : Node(ob) {}
  std::string typeString() const final { return "Reshape"; }
  std::unique_ptr<Node> clone(const State &) const final;
  DisjointRegions getInRegions(InIndex, const DisjointRegions &) const final;
  bool samples() const final { return false; }
  bool allocates() const final { return false; }
};

class Expand : public Node {
public:
  Expand(const State &ob) : Node(ob) {}
  std::string typeString() const final { return "Expand"; }
  std::unique_ptr<Node> clone(const State &) const final;
  DisjointRegions getInRegions(InIndex, const DisjointRegions &) const final;
  bool samples() const final { return false; }
  bool allocates() const final { return false; }
};

class Reverse : public Node {
public:
  Reverse(const State &ob, const std::vector<uint64_t> &d)
      : Node(ob), dims_(d) {}
  const std::vector<uint64_t> &dimensions() const { return dims_; }
  std::string typeString() const final;
  std::unique_ptr<Node> clone(const State &) const final;
  DisjointRegions getInRegions(InIndex, const DisjointRegions &) const final;
  bool samples() const final { return false; }
  bool allocates() const final { return false; }

private:
  const std::vector<uint64_t> dims_;
};

class Permute : public Node {
public:
  Permute(const State &ob, const Permutation &p_) : Node(ob), p(p_) {}
  const Permutation &permutation() const { return p; }
  std::string typeString() const final;
  std::unique_ptr<Node> clone(const State &) const final;
  DisjointRegions getInRegions(InIndex, const DisjointRegions &) const final;
  bool samples() const final { return false; }
  bool allocates() const final { return false; }

private:
  const Permutation p;
};

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
