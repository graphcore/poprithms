// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <ostream>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/transitiveclosureoptimizations.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

namespace {

std::array<std::string, NTCOS> initPmosNames() {
  std::array<std::string, NTCOS> names;
  for (auto &x : names) {
    x = "unset";
  }

  names[static_cast<uint64_t>(TransitiveClosureOptim::LinkTightDrops)] =
      "LinkTightDrops";

  names[static_cast<uint64_t>(TransitiveClosureOptim::LinkCloseTightPairs)] =
      "LinkCloseTightPairs";

  names[static_cast<uint64_t>(
      TransitiveClosureOptim::ConstrainWeightSeparatedGroups)] =
      "ConstrainWeightSeparatedGroups";

  names[static_cast<uint64_t>(
      TransitiveClosureOptim::ConstrainParallelChains)] =
      "ConstrainParallelChains";

  for (const auto &x : names) {
    if (x == "unset") {
      throw error("Failed to set all names for TCO enum");
    }
  }
  return names;
}

} // namespace

const std::array<std::string, NTCOS> &getPmosNames() {
  const auto static names = initPmosNames();
  return names;
}

std::ostream &operator<<(std::ostream &os, TransitiveClosureOptim tco) {
  os << getPmosNames()[static_cast<uint64_t>(tco)] << std::endl;
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const TransitiveClosureOptimizations &tcos) {
  const auto &names = getPmosNames();
  for (uint64_t i = 0; i < NTCOS; ++i) {
    os << names[i] << " : " << tcos.getVals()[i] << '\n';
  }
  return os;
}
} // namespace anneal
} // namespace schedule
} // namespace poprithms
