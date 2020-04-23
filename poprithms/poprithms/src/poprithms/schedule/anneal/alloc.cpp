#include <algorithm>

#include <poprithms/schedule/anneal/alloc.hpp>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/unisort.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

void Alloc::sortAndMakeUnique() { ops = util::unisorted(ops); }

void Alloc::appendSerialization(std::ostream &ost) const {
  ost << "{\"address\":" << address << ",\"weight\":";
  weight.appendSerialization(ost);
  ost << '}';
}

std::ostream &operator<<(std::ostream &ost, const Alloc &alloc) {
  ost << alloc.getAddress() << " ops=";
  util::append(ost, alloc.getOps());
  ost << " weight=" << alloc.getWeight();
  return ost;
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
