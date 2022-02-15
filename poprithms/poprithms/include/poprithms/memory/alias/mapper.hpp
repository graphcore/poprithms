// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_ALIAS_MAPPER_HPP
#define POPRITHMS_MEMORY_ALIAS_MAPPER_HPP

#include <memory>
#include <sstream>
#include <unordered_map>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/alias/tensor.hpp>
#include <poprithms/memory/alias/usings.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace memory {
namespace alias {

/**
 * A helper class for mapping between tensors in a memory::alias::Graph and
 * tensors in another (external) graph. The tensors in the external graph have
 * ids of template parameter type ExternTensorId.
 *
 * We assume that there are no duplicates in either direction. That is there
 * is at most 1 ExternTensorId for an alias TensorId, and at most 1 alias
 * TensorId for each ExternTensorId.
 *
 * This class owns a memory::alias::Graph, and maps between memory::alias
 * TensorIds and ExternTensorIds.
 * */
template <class ExternTensorId> class Mapper {
public:
  using ExternTensorIds = std::vector<ExternTensorId>;

  Mapper() = default;

  Mapper(const Mapper &) = default;
  Mapper(Mapper &&)      = default;

  Mapper &operator=(const Mapper &) = default;
  Mapper &operator=(Mapper &&) = default;

  virtual ~Mapper() = default;

  /**
   * For improved error messages, this virtual method must be implemented to
   * return the name/context of the external porject.
   * */
  virtual std::string external() const = 0;

  Graph &graph() { return graph_; }
  const Graph &graph() const { return graph_; }

  /** Get the unique alias graph tensor corresponding to #eId */
  TensorId id(const ExternTensorId &eId) const {
    auto found = toAlias_.find(eId);
    if (found == toAlias_.cend()) {
      std::ostringstream oss;
      oss << "Failed to find an alias TensorId for the " << external()
          << " TensorId " << eId << '.';
      throw poprithms::error::error("memory::alias", oss.str());
    }
    return found->second;
  }

  /** return true of #eId has an alias tensor corresponding to it. */
  bool has(const ExternTensorId &eId) const {
    return toAlias_.find(eId) != toAlias_.cend();
  }

  /** return true of #aId has an external tensor corresponding to it. */
  bool hasAliasId(const TensorId &aId) const {
    return toExtern_.find(aId) != toExtern_.cend();
  }

  /** Get the unique alias graph tensors corresponding to #eIds */
  TensorIds ids(const ExternTensorIds &eIds) const {
    TensorIds aIds;
    aIds.reserve(eIds.size());
    for (const auto &eId : eIds) {
      aIds.push_back(id(eId));
    }
    return aIds;
  }

  /**
   * Get the unique external tensor corresponding to the alias graph tensor
   * #aId
   * */
  ExternTensorId idFromAliasId(const TensorId &aId) const {
    auto found = toExtern_.find(aId);
    if (found == toExtern_.cend()) {
      std::ostringstream oss;
      oss << "Failed to find " << external()
          << " TensorId for the alias TensorId " << aId << '.';
      throw poprithms::error::error("memory::alias", oss.str());
    }
    return found->second;
  }

  /**
   * Get the unique external tensors corresponding to the alias graph tensors,
   * #aIds
   * */
  ExternTensorIds idsFromAliasIds(const TensorIds &aIds) const {
    ExternTensorIds eIds;
    eIds.reserve(aIds.size());
    for (const auto &aId : aIds) {
      eIds.push_back(idFromAliasId(aId));
    }
    return eIds;
  }

  Tensor tensor(const ExternTensorId &eId) { return graph().tensor(id(eId)); }

  Tensors tensors(const ExternTensorIds &eIds) {
    return graph().tensors(ids(eIds));
  }

  Tensor tensorFromAliasId(const TensorId &aId) {
    return graph().tensor(aId);
  }

  Tensors tensorsFromAliasIds(const TensorIds &aIds) {
    return graph().tensors(aIds);
  }

  /**
   * Register 1:1 mappings between alias graph tensors and external tensors.
   * #aIds and #eIds must be the same size, where aIds[i] corresponds to
   * eIds[i].
   * */
  void insert(const TensorIds &aIds, const ExternTensorIds &eIds) {

    // 1:1
    if (aIds.size() != eIds.size()) {
      std::ostringstream oss;
      oss << "Expected 1:1 correspondence between alias TensorIds ";
      poprithms::util::append(oss, aIds);
      oss << " and " << external() << " TensorIds ";
      poprithms::util::append(oss, eIds);
      oss << ", but the number of alias TensorIds (" << aIds.size()
          << ") != the number of " << external() << " TensorIds ("
          << eIds.size() << ").";
      throw poprithms::error::error("memory::alias", oss.str());
    }

    // all alias tensors are new (do not yet have corresponding external
    // tensors)
    for (const auto &aId : aIds) {
      auto found = toExtern_.find(aId);
      if (found != toExtern_.cend()) {
        std::ostringstream oss;
        oss << "Expected the alias TensorId " << aId
            << " to NOT be present in the map from alias TensorIds "
            << " to " << external() << " TensorIds, but it is, "
            << "mapping to " << external() << " TensorId " << found->second
            << ". External TensorIds should not share alias TensorIds. ";
        throw poprithms::error::error("memory::alias", oss.str());
      }
    }

    // all external tensors are new (do not yet have corresponding alias
    // tensors)
    for (const auto &eId : eIds) {
      auto found = toAlias_.find(eId);
      if (found != toAlias_.cend()) {
        std::ostringstream oss;
        oss << "Expected the " << external() << " TensorId " << eId
            << " to NOT be present in the map from alias TensorIds "
            << " to " << external() << " TensorIds, but it is, "
            << "mapping to " << external() << " TensorId " << found->second
            << ". Alias TensorIds should not share " << external()
            << " TensorIds. ";
        throw poprithms::error::error("memory::alias", oss.str());
      }
    }

    for (uint64_t i = 0; i < aIds.size(); ++i) {
      toExtern_.insert({aIds[i], eIds[i]});
      toAlias_.insert({eIds[i], aIds[i]});
    }
  }

private:
  Graph graph_;
  std::unordered_map<ExternTensorId, TensorId> toAlias_;
  std::unordered_map<TensorId, ExternTensorId> toExtern_;
};

} // namespace alias
} // namespace memory
} // namespace poprithms

#endif
