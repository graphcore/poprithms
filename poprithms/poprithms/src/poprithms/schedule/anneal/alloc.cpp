#include <algorithm>
#include <poprithms/schedule/anneal/alloc.hpp>
#include <poprithms/schedule/anneal/printiter.hpp>
#include <poprithms/schedule/anneal/unisort.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

void Alloc::sortAndMakeUnique() { ops = unisorted(ops); }

} // namespace anneal
} // namespace schedule
} // namespace poprithms
