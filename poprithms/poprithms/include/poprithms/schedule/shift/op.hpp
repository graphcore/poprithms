// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_OP_HPP
#define POPRITHMS_SCHEDULE_SHIFT_OP_HPP

#include <limits>
#include <sstream>
#include <vector>

#include <poprithms/schedule/shift/shiftusings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

/**
 * An Op has,
 * (1) an id (an OpAddress, which is an integer type)
 * (2) inputs and outputs (topological constraints with other Ops)
 * (3) optional forward and backward links (constraints to be contiguously
 *     scheduled with other Ops)
 * (4) a set of Allocs which must be live when it is scheduled
 * (5) a name (string)
 * */
class Op {

public:
  Op(OpAddress, const std::string &debugString);

  Op(const Op &) = default;
  Op(Op &&)      = default;

  Op &operator=(const Op &) = default;
  Op &operator=(Op &&) = default;

  ~Op() = default;
  Op()  = delete;

  void insertOut(OpAddress);
  void insertIn(OpAddress);
  void insertAlloc(AllocAddress);

  const std::vector<OpAddress> &getIns() const { return ins; }
  OpAddress getIn(uint64_t i) const { return ins[i]; }
  uint64_t nIns() const { return getIns().size(); }
  int nIns_i32() const { return static_cast<int>(nIns()); }
  bool hasIn(OpAddress a) const {
    return std::find(ins.cbegin(), ins.cend(), a) != ins.cend();
  }

  const std::vector<OpAddress> &getOuts() const { return outs; }
  OpAddress getOut(uint64_t i) const { return outs[i]; }
  uint64_t nOuts() const { return getOuts().size(); }
  int nOuts_i32() const { return static_cast<int>(nOuts()); }
  bool hasOut(OpAddress a) const {
    return std::find(outs.cbegin(), outs.cend(), a) != outs.cend();
  }

  const std::vector<AllocAddress> &getAllocs() const { return allocs; }
  AllocAddress getAlloc(uint64_t i) const { return allocs[i]; }
  uint64_t nAllocs() const { return getAllocs().size(); }
  bool hasAlloc(AllocAddress a) const {
    return std::find(allocs.cbegin(), allocs.cend(), a) != allocs.cend();
  }

  OpAddress getAddress() const { return address; }
  void append(std::ostream &ost) const;

  const std::string &getDebugString() const { return debugString; }

  bool hasForwardLink() const { return fwdLink != NoLinkVal; }
  bool hasBackwardLink() const { return bwdLink != NoLinkVal; }
  bool hasLink() const { return hasForwardLink() || hasBackwardLink(); }

  OpAddress getForwardLink() const { return fwdLink; }
  OpAddress getBackwardLink() const { return bwdLink; }

  bool operator==(const Op &rhs) const {
    return getFullComparitor() == rhs.getFullComparitor();
  }
  bool operator!=(const Op &rhs) const { return !operator==(rhs); }

  bool operator<(const Op &rhs) const {
    return getFullComparitor() < rhs.getFullComparitor();
  }

  void insertForwardLink(OpAddress after) { fwdLink = after; }
  void insertBackwardLink(OpAddress before) { bwdLink = before; }

  void removeIn(OpAddress i) {
    ins.erase(std::find(ins.cbegin(), ins.cend(), i));
  }

  void removeOut(OpAddress out) {
    outs.erase(std::find(outs.cbegin(), outs.cend(), out));
  }

  void appendSerialization(std::ostream &) const;

  using FullComparitor = std::tuple<OpAddress,
                                    std::vector<OpAddress>,
                                    std::vector<OpAddress>,
                                    std::vector<AllocAddress>,
                                    std::string,
                                    OpAddress,
                                    OpAddress>;

  using GraphComparitor = std::tuple<OpAddress,
                                     OpAddress,
                                     std::vector<OpAddress>,
                                     std::vector<AllocAddress>>;

  FullComparitor getFullComparitor() const {
    return {address, ins, outs, allocs, debugString, fwdLink, bwdLink};
  }

  GraphComparitor getGraphComparitor() const {
    return {address, fwdLink, outs, allocs};
  }

private:
  OpAddress address;
  std::vector<OpAddress> ins;
  std::vector<OpAddress> outs;
  std::vector<AllocAddress> allocs;
  std::string debugString;
  OpAddress fwdLink{NoLinkVal};
  OpAddress bwdLink{NoLinkVal};

public:
  static const OpAddress NoLinkVal = std::numeric_limits<OpAddress>::max();

}; // namespace shift

std::ostream &operator<<(std::ostream &ost, const Op &op);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
