#include <algorithm>
#include <poprithms/schedule/anneal/alloc.hpp>
#include <poprithms/schedule/anneal/printiter.hpp>
#include <poprithms/schedule/anneal/unisort.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

void Alloc::sortAndMakeUnique() { ops = unisorted(ops); }

std::ostream &operator<<(std::ostream &ost, const Alloc &alloc) {
  ost << alloc.getAddress() << " ops=";
  util::append(ost, alloc.getOps());
  ost << " weight=" << alloc.getWeight();
  return ost;
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
