// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <set>

#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/distributed/helper.hpp>
namespace poprithms {
namespace program {
namespace distributed {

using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;

SubGraphIds Helper::userReachable() const {

  // initialize a list of sub-graphs to process as the entry points to the
  // DAG: the user callable sub-graphs.
  SubGraphIds toProcess;
  std::set<SubGraphId> visited;
  for (auto x : userCallable()) {
    toProcess.push_back(x);
    visited.insert(x);
  }

  // Perform DFS on each of the sub-graphs in the queue. Edges arise from ops
  // with callees.
  SubGraphIds reachable;

  while (!toProcess.empty()) {
    auto nxt = toProcess.back();
    toProcess.pop_back();
    reachable.push_back(nxt);
    for (const auto &op : schedule(nxt)) {
      for (auto c : callees(op)) {
        if (visited.count(c) == 0) {
          visited.insert(c);
          toProcess.push_back(c);
        }
      }
    }
  }
  return reachable;
}

void Helper::noWeakVTables() {
  throw poprithms::error::error("distributed",
                                poprithms::error::error::weakVTableMessage());
}

} // namespace distributed
} // namespace program
} // namespace poprithms
