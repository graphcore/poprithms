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

template <typename T> std::vector<T> sorted(const std::vector<T> &x) {
  std::vector<T> y = x;
  std::sort(y.begin(), y.end());
  return y;
}

} // namespace

Op::Op(OpAddress _address_,
       const std::vector<OpAddress> &_ins_,
       const std::vector<AllocAddress> &_allocs_,
       const std::string &_debugString_)
    : address(_address_),       // address of this Op
      ins(sorted(_ins_)),       // addresses of input Ops
      allocs(sorted(_allocs_)), // addresses of this Op's Allocs
      debugString(_debugString_) {}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
