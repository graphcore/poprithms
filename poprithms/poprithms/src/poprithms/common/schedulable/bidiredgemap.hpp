// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_SCHEDULABLE_BIDIREDGEMAP_HPP
#define POPRITHMS_COMMON_SCHEDULABLE_BIDIREDGEMAP_HPP

#include <map>

#include <poprithms/common/multiout/opid.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

using common::multiout::OpId;
using common::multiout::OpIds;

/**
 * A bi-directional edge map.
 * */
class BiDirEdgeMap {

public:
  BiDirEdgeMap() = default;

  void insert(OpId from, OpId to);

  /**
   * All forward edges.
   * */
  std::vector<std::pair<OpId, OpId>> fwdEdges() const;

  /**
   * All backward edges.
   * */
  std::vector<std::pair<OpId, OpId>> bwdEdges() const;

  /**
   * \return All ops which are the start of an edge which terminates at #to.
   * */
  OpIds bwdEdges(OpId to) const;

  /**
   * \return All ops which are the end of an edge which starts at #from.
   * */
  OpIds fwdEdges(OpId from) const;

private:
  // forward edges
  std::map<OpId, OpIds> fwds;

  // backward edges (the reverse of forward edges)
  std::map<OpId, OpIds> bwds;
};

} // namespace schedulable
} // namespace common
} // namespace poprithms

#endif
