// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_SCHEDULABLE_ADDITIONALFWDEDGE_HPP
#define POPRITHMS_COMMON_SCHEDULABLE_ADDITIONALFWDEDGE_HPP

#include <poprithms/common/multiout/fwdedgemap.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

using common::multiout::FwdEdgeMap;
using common::multiout::OpId;
using common::multiout::OpIds;

/**
 * Base class for additional forward edges which can be passed to Graph
 * scheduling methods.
 * */
class AdditionalFwdEdges {
public:
  /**
   * \return all additional forward edges.
   * */
  virtual std::vector<std::pair<OpId, OpId>> fwdEdges() const = 0;

  /**
   * \return true if #opId is the source of an edge.
   * */
  virtual bool isSource(OpId opId) const = 0;

  /**
   * \return true if #from->#to is an edge.
   * */
  virtual bool isEdge(OpId from, OpId to) const = 0;

private:
  virtual void noWeakVTables();
};

class NoAdditionalFwdEdges : public AdditionalFwdEdges {
public:
  std::vector<std::pair<OpId, OpId>> fwdEdges() const final;
  bool isSource(OpId) const final { return false; }
  bool isEdge(OpId, OpId) const final { return false; }
};

/**
 * A map based forward edge class.
 * */
template <typename Map>
class AdditionalFwdEdgesFromMap : public AdditionalFwdEdges {
private:
  Map m;

public:
  AdditionalFwdEdgesFromMap(const Map &m_) : AdditionalFwdEdges(), m(m_) {}
  std::vector<std::pair<OpId, OpId>> fwdEdges() const final {
    std::vector<std::pair<OpId, OpId>> edges;
    for (const auto &f_ts : m) {
      for (auto t : f_ts.second) {
        edges.push_back({f_ts.first, t});
      }
    }
    return edges;
  }

  bool isSource(OpId f) const final { return m.find(f) != m.cend(); }

  bool isEdge(OpId from, OpId to) const final {
    auto iter = m.find(from);
    if (iter == m.cend()) {
      return false;
    }
    const auto &v = iter->second;
    return std::find(v.cbegin(), v.end(), to) != v.cend();
  }
};

} // namespace schedulable
} // namespace common
} // namespace poprithms

#endif
