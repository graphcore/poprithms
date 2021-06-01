// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_ALLOC_HPP
#define POPRITHMS_SCHEDULE_SHIFT_ALLOC_HPP

#include <tuple>
#include <vector>

#include <poprithms/schedule/shift/allocweight.hpp>
#include <poprithms/schedule/shift/shiftusings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

/**
 * An Alloc represents a memory allocation. It has
 * (1) a "size" represented by an AllocWeight
 * (2) an ID (an AllocAddress, which is an integer type)
 * (3) a set of Ops which require it to be live when they are scheduled.
 *
 * Allocs can also be used rocontrol the liveness of the Graph in non-memory
 * related ways.
 * */
class Alloc {

public:
  Alloc(AllocAddress a, AllocWeight w) : address(a), weight(w) {}

  Alloc(const Alloc &) = default;
  Alloc(Alloc &&)      = default;

  Alloc &operator=(const Alloc &) = default;
  Alloc &operator=(Alloc &&) = default;

  ~Alloc() = default;
  Alloc()  = delete;

  AllocAddress getAddress() const { return address; }
  AllocWeight getWeight() const { return weight; }

  // The Ops which require this Alloc to be live when they are scheduled
  void insertOp(OpAddress opAddress);
  const std::vector<OpAddress> &getOps() const { return ops; }
  size_t nOps() const { return getOps().size(); }
  int nOps_i32() const { return static_cast<int>(nOps()); }

  bool operator==(const Alloc &rhs) const {
    return getTuple() == rhs.getTuple();
  }

  bool operator<(const Alloc &rhs) const {
    return getTuple() < rhs.getTuple();
  }

  size_t hash() const;

  void append(std::ostream &) const;
  void appendSerialization(std::ostream &) const;

  std::tuple<AllocAddress, AllocWeight, std::vector<OpAddress>>
  getTuple() const {
    return {address, weight, ops};
  }

private:
  AllocAddress address;

  // The weight should be proportional to the number of bytes used
  AllocWeight weight;
  std::vector<OpAddress> ops;
};

std::size_t hash_value(const Alloc &);

std::ostream &operator<<(std::ostream &ost, const Alloc &alloc);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
