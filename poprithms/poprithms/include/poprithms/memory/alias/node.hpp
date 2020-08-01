// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_OP_HPP
#define POPRITHMS_MEMORY_ALIAS_OP_HPP

#include <map>
#include <memory>
#include <sstream>

#include <poprithms/memory/alias/origins.hpp>
#include <poprithms/memory/alias/tensor.hpp>
#include <poprithms/memory/gbase/node.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/shape.hpp>

namespace poprithms {
namespace memory {
namespace alias {

/**
 * Extend the gbase::Node with aliasing information.
 * */
class Node : public gbase::Node {

public:
  using State = gbase::Node::State;

  virtual ~Node() = default;

  Node(const State &ob, const Origins &oris)
      : gbase::Node(ob), origins_(oris) {}

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
  Origins origins_;
};

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
