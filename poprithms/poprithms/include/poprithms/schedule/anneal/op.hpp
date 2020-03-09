#ifndef POPRITHMS_SCHEDULE_ANNEAL_OP_HPP
#define POPRITHMS_SCHEDULE_ANNEAL_OP_HPP

#include <limits>
#include <sstream>
#include <vector>
#include <poprithms/schedule/anneal/annealusings.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

class Op {

public:
  Op(OpAddress _address_, const std::string &_debugString_);

  void insertOut(OpAddress out) { outs.push_back(out); }
  void insertIn(OpAddress i) { ins.push_back(i); }
  void insertAlloc(AllocAddress aa) { allocs.push_back(aa); }

  const std::vector<OpAddress> &getIns() const { return ins; }
  OpAddress getIn(uint64_t i) const { return ins[i]; }
  uint64_t nIns() const { return getIns().size(); }
  int nIns_i32() const { return static_cast<int>(nIns()); }

  bool hasIn(OpAddress a) const {
    return std::find(ins.cbegin(), ins.cend(), a) != ins.cend();
  }

  bool hasOut(OpAddress a) const {
    return std::find(outs.cbegin(), outs.cend(), a) != outs.cend();
  }

  const std::vector<OpAddress> &getOuts() const { return outs; }
  OpAddress getOut(uint64_t i) const { return outs[i]; }
  uint64_t nOuts() const { return getOuts().size(); }
  int nOuts_i32() const { return static_cast<int>(nOuts()); }

  const std::vector<AllocAddress> &getAllocs() const { return allocs; }
  AllocAddress getAlloc(uint64_t i) const { return allocs[i]; }
  uint64_t nAllocs() const { return getAllocs().size(); }
  bool hasAlloc(AllocAddress a) const {
    return std::find(allocs.cbegin(), allocs.cend(), a) != allocs.cend();
  }

  OpAddress getAddress() const { return address; }
  void append(std::ostream &ost) const;

  const std::string &getDebugString() const { return debugString; }

  void sortAndMakeUnique();

  bool hasForwardLink() const { return fwdLink != NoLinkVal; }
  bool hasBackwardLink() const { return bwdLink != NoLinkVal; }
  bool hasLink() const { return hasForwardLink() || hasBackwardLink(); }

  OpAddress getForwardLink() const { return fwdLink; }
  OpAddress getBackwardLink() const { return bwdLink; }

  bool operator==(const Op &rhs) const {
    return address == rhs.address && ins == rhs.ins && outs == rhs.outs &&
           allocs == rhs.allocs && debugString == rhs.debugString;
  }
  bool operator!=(const Op &rhs) const { return !operator==(rhs); }

  void insertForwardLink(OpAddress after) { fwdLink = after; }
  void insertBackwardLink(OpAddress before) { bwdLink = before; }

  void removeIn(OpAddress i) {
    ins.erase(std::find(ins.cbegin(), ins.cend(), i));
  }

  void removeOut(OpAddress out) {
    outs.erase(std::find(outs.cbegin(), outs.cend(), out));
  }

  void appendSerialization(std::ostream &) const;

private:
  const OpAddress address;
  std::vector<OpAddress> ins;
  std::vector<OpAddress> outs;
  std::vector<AllocAddress> allocs;
  const std::string debugString;
  OpAddress fwdLink{NoLinkVal};
  OpAddress bwdLink{NoLinkVal};

public:
  static const OpAddress NoLinkVal = std::numeric_limits<OpAddress>::max();

}; // namespace anneal

std::ostream &operator<<(std::ostream &ost, const Op &op);

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
