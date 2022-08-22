// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <array>
#include <limits>
#include <ostream>

#include <schedule/shift/error.hpp>

#include <poprithms/schedule/shift/transitiveclosureoptimizations.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

void TransitiveClosureOptimizations::Option::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

TransitiveClosureOptim
TransitiveClosureOptimizations::LinkTightDrops::getEnum() const {
  return TransitiveClosureOptim::LinkTightDrops;
}

TransitiveClosureOptim
TransitiveClosureOptimizations::LinkCloseTightPairs::getEnum() const {
  return TransitiveClosureOptim::LinkCloseTightPairs;
}

TransitiveClosureOptim
TransitiveClosureOptimizations::ConstrainWeightSeparatedGroups::getEnum()
    const {
  return TransitiveClosureOptim::ConstrainWeightSeparatedGroups;
}

TransitiveClosureOptim
TransitiveClosureOptimizations::ConstrainParallelChains::getEnum() const {
  return TransitiveClosureOptim::ConstrainParallelChains;
}

TransitiveClosureOptim
TransitiveClosureOptimizations::CombineAllocsWithCommonOps::getEnum() const {
  return TransitiveClosureOptim::CombineAllocsWithCommonOps;
}

TransitiveClosureOptim
TransitiveClosureOptimizations::DisconnectAllocsWithOneOp::getEnum() const {
  return TransitiveClosureOptim::DisconnectAllocsWithOneOp;
}

TransitiveClosureOptim
TransitiveClosureOptimizations::DisconnectAllocsWithZeroWeight::getEnum()
    const {
  return TransitiveClosureOptim::DisconnectAllocsWithZeroWeight;
}

TransitiveClosureOptim
TransitiveClosureOptimizations::DisconnectInbetweenerAllocs::getEnum() const {
  return TransitiveClosureOptim::DisconnectInbetweenerAllocs;
}

TransitiveClosureOptim
TransitiveClosureOptimizations::DisconnectFixedDurationAllocs::getEnum()
    const {
  return TransitiveClosureOptim::DisconnectFixedDurationAllocs;
}

TransitiveClosureOptim
TransitiveClosureOptimizations::ConnectContiguousAllocs::getEnum() const {
  return TransitiveClosureOptim::ConnectContiguousAllocs;
}

std::ostream &operator<<(std::ostream &os,
                         const TransitiveClosureOptimizations &tcos) {
  tcos.append(os);
  return os;
}

bool TransitiveClosureOptimizations::operator!=(
    const TransitiveClosureOptimizations &rhs) const {
  return !operator==(rhs);
}

void TransitiveClosureOptimizations::append(std::ostream &os) const {
  for (auto o : getOptions()) {
    os << "   ";
    o->append(os);
    os << '\n';
  }
}

bool TransitiveClosureOptimizations::allOptimizationsOn() const {
  auto opts = getOptions();
  return std::all_of(
      opts.cbegin(), opts.cend(), [](const auto &o) { return o->on(); });
}

bool TransitiveClosureOptimizations::allOptimizationsOff() const {
  auto opts = getOptions();
  return std::all_of(
      opts.cbegin(), opts.cend(), [](const auto &o) { return o->off(); });
}

TransitiveClosureOptimizations TransitiveClosureOptimizations::all(bool a) {
  return TransitiveClosureOptimizations(LinkTightDrops(a),
                                        LinkCloseTightPairs(a),
                                        ConstrainWeightSeparatedGroups(a),
                                        ConstrainParallelChains(a),
                                        CombineAllocsWithCommonOps(a),
                                        DisconnectAllocsWithOneOp(a),
                                        DisconnectAllocsWithZeroWeight(a),
                                        DisconnectInbetweenerAllocs(a),
                                        DisconnectFixedDurationAllocs(a),
                                        ConnectContiguousAllocs(a),
                                        std::numeric_limits<int>::max());
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withConstrainParallelChains(bool on__) {
  constrainParallelChains_.update(on__);
  return *this;
}
bool TransitiveClosureOptimizations::constrainParallelChains() const {
  return constrainParallelChains_.on();
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withLinkTightDrops(bool on__) {
  linkTightDrops_.update(on__);
  return *this;
}
bool TransitiveClosureOptimizations::linkTightDrops() const {
  return linkTightDrops_.on();
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withLinkCloseTightPairs(bool on__) {
  linkCloseTightPairs_.update(on__);
  return *this;
}
bool TransitiveClosureOptimizations::linkCloseTightPairs() const {
  return linkCloseTightPairs_.on();
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withConstrainWeightSeparatedGroups(
    bool on__) {
  constrainWeightSeparatedGroups_.update(on__);
  return *this;
}
bool TransitiveClosureOptimizations::constrainWeightSeparatedGroups() const {
  return constrainWeightSeparatedGroups_.on();
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withCombineAllocsWithCommonOps(bool on__) {
  combineAllocsWithCommonOps_.update(on__);
  return *this;
}
bool TransitiveClosureOptimizations::combineAllocsWithCommonOps() const {
  return combineAllocsWithCommonOps_.on();
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withDisconnectAllocsWithOneOp(bool on__) {
  disconnectAllocsWithOneOp_.update(on__);
  return *this;
}
bool TransitiveClosureOptimizations::disconnectAllocsWithOneOp() const {
  return disconnectAllocsWithOneOp_.on();
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withDisconnectAllocsWithZeroWeight(
    bool on__) {
  disconnectAllocsWithZeroWeight_.update(on__);
  return *this;
}
bool TransitiveClosureOptimizations::disconnectAllocsWithZeroWeight() const {
  return disconnectAllocsWithZeroWeight_.on();
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withDisconnectInbetweenerAllocs(bool on__) {
  disconnectInbetweenerAllocs_.update(on__);
  return *this;
}
bool TransitiveClosureOptimizations::disconnectInbetweenerAllocs() const {
  return disconnectInbetweenerAllocs_.on();
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withDisconnectFixedDurationAllocs(bool on__) {
  disconnectFixedDurationAllocs_.update(on__);
  return *this;
}
bool TransitiveClosureOptimizations::disconnectFixedDurationAllocs() const {
  return disconnectFixedDurationAllocs_.on();
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withConnectContiguousAllocs(bool on__) {
  connectContiguousAllocs_.update(on__);
  return *this;
}
bool TransitiveClosureOptimizations::connectContiguousAllocs() const {
  return connectContiguousAllocs_.on();
}

std::vector<TransitiveClosureOptim>
TransitiveClosureOptimizations::enabled() const {
  std::vector<TransitiveClosureOptim> optims;

  for (auto opt : getOptions()) {
    if (opt->on()) {
      optims.push_back(opt->getEnum());
    }
  }

  if (slideLinks()) {
    optims.push_back(TransitiveClosureOptim::SlideLinks);
  }

  return optims;
}

bool TransitiveClosureOptimizations::operator<(
    const TransitiveClosureOptimizations &rhs) const {
  return enabled() < rhs.enabled();
}

TransitiveClosureOptimizations TransitiveClosureOptimizations::allOn() {
  return all(true);
}

bool TransitiveClosureOptimizations::operator==(
    const TransitiveClosureOptimizations &rhs) const {
  return enabled() == rhs.enabled();
}

TransitiveClosureOptimizations &
TransitiveClosureOptimizations::withMaxIterations(int n) {
  maxNumberOfIterations = n;
  return *this;
}

std::string
TransitiveClosureOptimizations::str(TransitiveClosureOptim optim) {

  switch (optim) {
  case TransitiveClosureOptim::SlideLinks: {
    return "SlideLinks";
  }
  case TransitiveClosureOptim::LinkTightDrops: {
    return "LinkTightDrops";
  }
  case TransitiveClosureOptim::LinkCloseTightPairs: {
    return "LinkCloseTightPairs";
  }
  case TransitiveClosureOptim::ConstrainWeightSeparatedGroups: {
    return "ConstrainWeightSeparatedGroups";
  }
  case TransitiveClosureOptim::ConstrainParallelChains: {
    return "ConstrainParallelChains";
  }
  case TransitiveClosureOptim::CombineAllocsWithCommonOps: {
    return "CombineAllocsWithCommonOps";
  }
  case TransitiveClosureOptim::DisconnectAllocsWithOneOp: {
    return "DisconnectAllocsWithOneOp";
  }
  case TransitiveClosureOptim::DisconnectAllocsWithZeroWeight: {
    return "DisconnectAllocsWithZeroWeight";
  }
  case TransitiveClosureOptim::DisconnectInbetweenerAllocs: {
    return "DisconnectInbetweenerAllocs";
  }
  case TransitiveClosureOptim::DisconnectFixedDurationAllocs: {
    return "DisconnectFixedDurationAllocs";
  }
  case TransitiveClosureOptim::ConnectContiguousAllocs: {
    return "ConnectContiguousAllocs";
  }

  case TransitiveClosureOptim::N: {
    throw error("N is not a TransitiveClosureOptim with a string");
  }
  }

  throw error("No case for this TransitiveClosureOptim in 'str'");
}

std::ostream &operator<<(std::ostream &os, const TransitiveClosureOptim &o) {
  os << TransitiveClosureOptimizations::str(o);
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const std::vector<TransitiveClosureOptim> &o) {
  poprithms::util::append(os, o);
  return os;
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
