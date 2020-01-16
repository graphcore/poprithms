#ifndef POPRITHMS_SCHEDULE_ANNEAL_ALLOC_HPP
#define POPRITHMS_SCHEDULE_ANNEAL_ALLOC_HPP

#include <vector>
#include <poprithms/schedule/anneal/allocweight.hpp>
#include <poprithms/schedule/anneal/annealusings.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

// A memory allocation.
class Alloc {

public:
  Alloc(AllocAddress a, AllocWeight w) : address(a), weight(w) {}
  AllocAddress getAddress() const { return address; }
  AllocWeight getWeight() const { return weight; }

  // The Ops which require this Alloc to be live when they are scheduled
  void insertOp(OpAddress opAddress) { ops.push_back(opAddress); }
  const std::vector<OpAddress> &getOps() const { return ops; }
  size_t nOps() const { return getOps().size(); }
  int nOps_i32() const { return static_cast<int>(nOps()); }

  bool operator==(const Alloc &rhs) const {
    return address == rhs.address && weight == rhs.weight && ops == rhs.ops;
  }

private:
  const AllocAddress address;

  // The weight should be proportional to the number of bytes used
  const AllocWeight weight;
  std::vector<OpAddress> ops;
};

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
