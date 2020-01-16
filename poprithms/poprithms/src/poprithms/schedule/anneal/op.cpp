#include <algorithm>
#include <poprithms/schedule/anneal/op.hpp>
#include <poprithms/schedule/anneal/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

std::ostream &operator<<(std::ostream &ost, const Op &op) {
  op.append(ost);
  return ost;
}

void Op::append(std::ostream &ost) const { ost << debugString; }

namespace {

template <typename T> std::vector<T> unisorted(const std::vector<T> &x) {
  std::vector<T> y = x;
  std::sort(y.begin(), y.end());
  auto last = std::unique(y.begin(), y.end());
  y.erase(last, y.cend());
  return y;
}

} // namespace

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
