// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstring>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include <poprithms/memory/inplace/error.hpp>
#include <poprithms/memory/inplace/result.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

std::ostream &operator<<(std::ostream &ost, const Constraints &constraints) {
  std::vector<std::string> frags;
  for (const auto &[from, to] : constraints) {
    frags.push_back("(" + from + "->" + to + ")");
  }
  poprithms::util::append(ost, frags);
  return ost;
}

void InplaceResult::append(std::ostream &ost) const {
  ost << status();
  if (status() == InplaceStatus::Valid) {
    ost << "(";
    ost << constraints() << ")";
  }
  ost << "scheduleChange=" << (scheduleChange_ ? "Yes" : "No");
}

std::ostream &operator<<(std::ostream &ost, const InplaceResult &r) {
  r.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const InplaceResults &rs) {
  if (rs.empty()) {
    ost << "()";
    return ost;
  }

  ost << '(' << rs[0];
  for (uint64_t i = 1; i < rs.size(); ++i) {
    ost << ',' << rs[i];
  }
  ost << ')';
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const InplaceStatus &s) {

  switch (s) {
  case InplaceStatus::AlreadyInplace: {
    ost << "AlreadyInplace";
    break;
  }
  case InplaceStatus::Valid: {
    ost << "Valid";
    break;
  }
  case InplaceStatus::Cycle: {
    ost << "Cycle";
    break;
  }
  case InplaceStatus::NotParallelWriteable: {
    ost << "NotParallelWriteable";
    break;
  }
  default: {
    throw error("Unrecognised Status");
  }
  }

  return ost;
}

const Constraints &InplaceResult::constraints() const {
  if (status() != InplaceStatus::Valid) {
    std::ostringstream oss;
    oss << "Call to InplaceResults::constraints, on " << *this
        << ". This method is only valid for InplaceResults with "
        << "InplaceStatus::Valid status. ";
    throw error(oss.str());
  }
  return constraints_;
}

const OpIds &InplaceResult::schedule() const {
  if (status() != InplaceStatus::Valid) {
    std::ostringstream oss;
    oss << "Call to InplaceResults::schedule, on " << *this
        << ". This method is only valid for InplaceResults with "
        << "a changed schedule, non-changed schedules are not stored. ";
    throw error(oss.str());
  }
  return schedule_;
}

InplaceResult InplaceResult::validWithUnchangedSchedule(Constraints &&cs) {
  return InplaceResult(InplaceStatus::Valid, std::move(cs), {}, false);
}
InplaceResult InplaceResult::validWithChangedSchedule(Constraints &&cs,
                                                      OpIds &&sched) {
  return InplaceResult(
      InplaceStatus::Valid, std::move(cs), std::move(sched), true);
}

InplaceResult InplaceResult::cycle() {
  return InplaceResult(InplaceStatus::Cycle, {}, {}, false);
}

InplaceResult InplaceResult::alreadyInplace() {
  return InplaceResult(InplaceStatus::AlreadyInplace, {}, {}, false);
}

InplaceResult InplaceResult::notParallelWriteable() {
  return InplaceResult(InplaceStatus::NotParallelWriteable, {}, {}, false);
}

std::ostream &operator<<(std::ostream &ost, const InplaceStatuses &statuses) {
  ost << '(';
  if (!statuses.empty()) {
    ost << statuses[0];
  }
  for (uint64_t i = 1; i < statuses.size(); ++i) {
    ost << ',' << statuses[i];
  }
  ost << ')';
  return ost;
}

} // namespace inplace
} // namespace memory
} // namespace poprithms
