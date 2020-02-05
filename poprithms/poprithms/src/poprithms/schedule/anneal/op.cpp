#include <algorithm>
#include <poprithms/schedule/anneal/op.hpp>
#include <poprithms/schedule/anneal/printiter.hpp>
#include <poprithms/schedule/anneal/unisort.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

std::ostream &operator<<(std::ostream &ost, const Op &op) {
  op.append(ost);
  return ost;
}

void Op::append(std::ostream &ost) const { ost << debugString; }

void Op::sortAndMakeUnique() {
  ins    = unisorted(ins);
  outs   = unisorted(outs);
  allocs = unisorted(allocs);
}

Op::Op(OpAddress _address_, const std::string &_debugString_)
    : address(_address_), debugString(_debugString_) {}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
