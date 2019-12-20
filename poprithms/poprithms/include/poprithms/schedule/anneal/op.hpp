#include <sstream>
#include <vector>
#include <poprithms/schedule/anneal/annealusings.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

class Op {

public:
  Op(OpAddress _address_,
     const std::vector<OpAddress> &_ins_,
     const std::vector<AllocAddress> &_allocs_,
     const std::string &_debugString_);

  void insertOut(OpAddress out) { outs.push_back(out); }

  const std::vector<OpAddress> &getIns() const { return ins; }
  uint64_t nIns() const { return getIns().size(); }

  const std::vector<OpAddress> &getOuts() const { return outs; }
  uint64_t nOuts() const { return getOuts().size(); }

  const std::vector<AllocAddress> &getAllocs() const { return allocs; }
  uint64_t nAllocs() const { return getAllocs().size(); }

  OpAddress getAddress() const { return address; }
  void append(std::ostream &ost) const;

  const std::string &getDebugString() const { return debugString; }

private:
  const OpAddress address;
  const std::vector<OpAddress> ins;
  std::vector<OpAddress> outs;
  const std::vector<AllocAddress> allocs;
  const std::string debugString;

}; // namespace anneal

std::ostream &operator<<(std::ostream &ost, const Op &op);

} // namespace anneal
} // namespace schedule
} // namespace poprithms
