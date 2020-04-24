#ifndef POPRITHMS_SCHEDULE_SUPER_GRAPH
#define POPRITHMS_SCHEDULE_SUPER_GRAPH

#include <array>
#include <vector>

namespace poprithms {
namespace schedule {
namespace supercon {

using OpId   = uint64_t;
using Edges  = std::vector<std::vector<OpId>>;
using Pair   = std::array<OpId, 2>;
using Arrows = std::array<std::array<uint64_t, 2>, 4>;

// First-in last-out (Filo) Kahn's algorithm with super constraints.
//
//
// Input arguments:
//
// 1) forwardEdges :
// -----------------
// the standard topological constraints of a DAG, that is,
// b \in edges[a] implies that b appears before a in the schedule.
//
// 2) couples :
// ------------
// the constraint here is that for all v \in couples,
//   v[0] before v[1] if and only if v[2] before v[3].
//
// As an example, suppose the Graph is
//
//    A   E
//   /|   |\
//  B C   F G
//   \|   |/
//    D   H
//
// and the {B,C,F,G} \in couples.
//
// The valid schedules are with this couple pair are:
// ABCDEFGH
// EFGHABCD
// ACBDEGFH
// EGFHACBD .
//
// In other words, valid schedules have (B before C) == (F before G).
//
// Note that {B,C,F,G}, {C,B,G,F}, {F,G,B,C} and {G,F,C,B} are equivalent.
//
//
// 3) bins :
// ---------
// coming soon, see TODO(T19634)
//
std::vector<uint64_t>
getFiloSchedule(const Edges &forwardEdges,
                const std::vector<std::array<OpId, 4>> &couples
                /* bins: (T19634) */
);

} // namespace supercon
} // namespace schedule
} // namespace poprithms

#endif
