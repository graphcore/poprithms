#ifndef POPRITHMS_OUTLINE_LINEAR_LINEARUSINGS_HPP
#define POPRITHMS_OUTLINE_LINEAR_LINEARUSINGS_HPP

#include <cstdint>
#include <vector>

#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace outline {
namespace linear {

// Why is the namespace called "linear"? All algorithms in this namespace
// linearize (a.k.a. topologically sort, or schedule) Graphs, and only
// considers sub-graphs which are sub-sequences in the schedule

// Notation:
// A Match consists of several equivalent Subgraphs. Each Match can be
//   -----                     ---------- ---------       -----
// converted to a single CallOp, and each Subgraph in the Match corresponds to
//                       ------           --------        -----
// a call of the CallOp.
//               ------

using Shape = std::vector<uint64_t>;

enum class DType { INT32 = 0, FLOAT16, FLOAT32, N };

std::ostream &operator<<(std::ostream &, DType);

// 2 Ops in corresponding positions in 2 equivalent Subgraphs must have the
// same Type. Type can correspond to <OpType, OpParams, DeviceId> for example.
// It is a frameworks task to determine how to partition Ops into Types.
using Type = int;

using TensorId = util::TypedInteger<'T', uint64_t>;
using OpId     = util::TypedInteger<'O', uint64_t>;

// 2 Ops in the same Subgraph of a Match must have the same Color. 2 Ops
// in different Subgraphs of a Match can have different Colors. In other
// words, a Subgraph which is non-monochromatic is not equivalent to any
// Subgraph.
//
// Use cases in PopArt:
// - ensure no-outlining across PingPong phases
// - ensure no outlining across recompute-checkpoint boundaries
using Color = util::TypedInteger<'C', int>;

using InIndex       = uint64_t;
using OutIndex      = uint64_t;
using ScheduleIndex = uint64_t;

} // namespace linear
} // namespace outline
} // namespace poprithms

#endif
