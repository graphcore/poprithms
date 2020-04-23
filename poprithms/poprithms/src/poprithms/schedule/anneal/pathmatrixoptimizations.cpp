#include <array>
#include <ostream>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/pathmatrixoptimizations.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

namespace {

std::array<std::string, NPMOS> initPmosNames() {
  std::array<std::string, NPMOS> names;
  for (auto &x : names) {
    x = "unset";
  }

  names[static_cast<uint64_t>(PMO::LinkTightDrops)] = "LinkTightDrops";

  names[static_cast<uint64_t>(PMO::LinkCloseTightPairs)] =
      "LinkCloseTightPairs";

  names[static_cast<uint64_t>(PMO::ConstrainWeightSeparatedGroups)] =
      "ConstrainWeightSeparatedGroups";

  names[static_cast<uint64_t>(PMO::ConstrainParallelChains)] =
      "ConstrainParallelChains";

  for (const auto &x : names) {
    if (x == "unset") {
      throw error("Failed to set all names for PMO enum");
    }
  }
  return names;
}

} // namespace

const std::array<std::string, NPMOS> &getPmosNames() {
  const auto static names = initPmosNames();
  return names;
}

std::ostream &operator<<(std::ostream &os, PMO pmo) {
  os << getPmosNames()[static_cast<uint64_t>(pmo)] << std::endl;
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const PathMatrixOptimizations &pmos) {
  const auto &names = getPmosNames();
  for (uint64_t i = 0; i < NPMOS; ++i) {
    os << names[i] << " : " << pmos.getVals()[i] << '\n';
  }
  return os;
}
} // namespace anneal
} // namespace schedule
} // namespace poprithms
