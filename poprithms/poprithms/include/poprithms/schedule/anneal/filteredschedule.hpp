#ifndef POPRITHMS_SCHEDULE_ANNEAL_FILTEREDSCHEDULE_GRAPH
#define POPRITHMS_SCHEDULE_ANNEAL_FILTEREDSCHEDULE_GRAPH

#include <unordered_map>
#include <poprithms/schedule/anneal/graph.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

// Starting at Op "id0", perform Kahn's algorithm on Graph "g", but block on
// all Ops "x" for which allow(x) is false.
//
// Examples:
//
//    0
//   / \
//  1   2
//  |   |\
//  3   4 6
//   \ / \|
//    5   7
//
//  id0  allow           returned:
// -------------------------------
//  0    {true}          all Ops
//  1    {true}          {1,3}
//  1    {less than 3}   {1}
//  2    {not 4}         {2, 6}
//  2    {true}          {2, 4, 6, 7}
//

template <typename Filter>
std::vector<OpAddress>
getFilteredSchedule(const Graph &g, OpAddress id0, const Filter &allow) {
  std::vector<OpAddress> toProcess{id0};
  std::vector<OpAddress> sched;
  std::unordered_map<OpAddress, int> outstanding;
  while (!toProcess.empty()) {
    auto current = toProcess.back();
    toProcess.pop_back();
    if (allow(current)) {
      sched.push_back(current);
      auto outs = g.getOp(current).getOuts();
      for (auto o : outs) {
        if (g.getOp(o).nIns() == 1) {
          toProcess.push_back(o);
        } else {
          auto found = outstanding.find(o);
          if (found == outstanding.cend()) {
            outstanding[o] = g.getOp(o).nIns_i32() - 1;
          } else {
            found->second -= 1;
            if (found->second == 0) {
              toProcess.push_back(o);
            }
          }
        }
      }
    } else {
      // current is not allowed. This means it will not be scheduled, and
      // nothing downstream of it will be scheduled.
    }
  }
  return sched;
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
