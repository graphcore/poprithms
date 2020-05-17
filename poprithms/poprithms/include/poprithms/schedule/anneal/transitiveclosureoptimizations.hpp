#ifndef POPRITHMS_SCHEDULE_ANNEAL_TRANSITIVECLOSUREOPTIMIZATIONS
#define POPRITHMS_SCHEDULE_ANNEAL_TRANSITIVECLOSUREOPTIMIZATIONS

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace poprithms {
namespace schedule {
namespace anneal {

// Enumaration of all the currently supported optimizations
enum class TransitiveClosureOptim {
  LinkTightDrops = 0,
  LinkCloseTightPairs,
  ConstrainWeightSeparatedGroups,
  ConstrainParallelChains,
  N // this is not an optimization, it is the number of optimizations
};

// Brief descriptions
// ------------------
//
// LinkTightDrops: if (a,b) is a tight pair, and b is guaranteed to increase
// liveness less than a, then upgrade (a,b) to a linked pair.
//
// LinkCloseTightPairs: if (a,b) is a tight pair, and there is no Op c in the
// unconstrained dual of a which can have an in increase in liveness equal to
// or between those of a and b, then upgrade (a,b) to a linked pair.
//
// ConstrainWeightSeparatedGroups: If a and b have common inputs, and there is
// guaranteed to the increases in livenesses in PostUnconstrained(a,b) are all
// less than or equal to those in PostUnconstrained(b,a), then insert a
// constraint a->b and some related constraints
//
// ConstrainParallelChains: If a and b have common inputs, and both belong to
// tight chains with common inputs, and if (1) a's chain is not shorter than
// b's and (2) the cumulative increase in liveness along a's chain is never
// greater than along b's, the insert constrains from a's chain to b's chain,
// to form a ladder of constraints
//
// Proofs of global optimality are currently being worked on.
//

std::ostream &operator<<(std::ostream &, TransitiveClosureOptim);
constexpr uint64_t NTCOS = static_cast<uint64_t>(TransitiveClosureOptim::N);

class TransitiveClosureOptimizations {

public:
  TransitiveClosureOptimizations(const std::array<bool, NTCOS> &x, int mits)
      : vals(x), maxNumberOfIterations(mits) {}

  TransitiveClosureOptimizations()
      : TransitiveClosureOptimizations({0, 0, 0, 0},
                                       std::numeric_limits<int>::max()) {}

  static TransitiveClosureOptimizations allOff() {
    return TransitiveClosureOptimizations();
  }

  bool allOptimizationsOff() const {
    return std::all_of(
        vals.cbegin(), vals.cend(), [](bool b) { return b == 0; });
  }

  static TransitiveClosureOptimizations allOn() {
    TransitiveClosureOptimizations tco;
    std::fill(tco.vals.begin(), tco.vals.end(), 1);
    return tco;
  }

  TransitiveClosureOptimizations &withConstrainParallelChains(bool b = true) {
    return update(TransitiveClosureOptim::ConstrainParallelChains, b);
  }
  bool constrainParallelChains() const {
    return at(TransitiveClosureOptim::ConstrainParallelChains);
  }

  TransitiveClosureOptimizations &withLinkTightDrops(bool b = true) {
    return update(TransitiveClosureOptim::LinkTightDrops, b);
  }
  bool linkTightDrops() const {
    return at(TransitiveClosureOptim::LinkTightDrops);
  }

  TransitiveClosureOptimizations &withLinkCloseTightPairs(bool b = true) {
    return update(TransitiveClosureOptim::LinkCloseTightPairs, b);
  }
  bool linkCloseTightPairs() const {
    return at(TransitiveClosureOptim::LinkCloseTightPairs);
  }

  TransitiveClosureOptimizations &
  withConstrainWeightSeparatedGroups(bool b = true) {
    return update(TransitiveClosureOptim::ConstrainWeightSeparatedGroups, b);
  }
  bool constrainWeightSeparatedGroups() const {
    return at(TransitiveClosureOptim::ConstrainWeightSeparatedGroups);
  }

  TransitiveClosureOptimizations &withMaxIterations(int mits) {
    maxNumberOfIterations = mits;
    return *this;
  }
  int maxIterations() const { return maxNumberOfIterations; }

  const std::array<bool, NTCOS> &getVals() const { return vals; }

private:
  bool at(TransitiveClosureOptim tco) const {
    return vals[static_cast<uint64_t>(tco)];
  }
  TransitiveClosureOptimizations &update(TransitiveClosureOptim tco, bool b) {
    vals[static_cast<uint64_t>(tco)] = (b == false ? 0 : 1);
    return *this;
  }

  std::array<bool, NTCOS> vals;
  int maxNumberOfIterations;
};

const std::array<std::string, NTCOS> &getPmosNames();
std::ostream &operator<<(std::ostream &os,
                         const TransitiveClosureOptimizations &);

} // namespace anneal
} // namespace schedule
} // namespace poprithms

#endif
