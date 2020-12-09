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

void OpeningResult::append(std::ostream &ost) const {
  ost << status();
  if (status() == OpeningStatus::Valid) {
    ost << "(";
    ost << constraints() << ")";
  }
  ost << "scheduleChange=" << (scheduleChange_ ? "Yes" : "No");
}

std::ostream &operator<<(std::ostream &ost, const OpeningResult &r) {
  r.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const OpeningResults &rs) {
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

std::ostream &operator<<(std::ostream &ost, const OpeningStatus &s) {

  switch (s) {
  case OpeningStatus::AlreadyOpen: {
    ost << "AlreadyInplace";
    break;
  }
  case OpeningStatus::Valid: {
    ost << "Valid";
    break;
  }
  case OpeningStatus::Cycle: {
    ost << "Cycle";
    break;
  }
  case OpeningStatus::NotParallelWriteable: {
    ost << "NotParallelWriteable";
    break;
  }
  default: {
    throw error("Unrecognised Status");
  }
  }

  return ost;
}

const Constraints &OpeningResult::constraints() const {
  if (status() != OpeningStatus::Valid) {
    std::ostringstream oss;
    oss << "Call to OpeningResults::constraints, on " << *this
        << ". This method is only valid for OpeningResults with "
        << "OpeningStatus::Valid status. ";
    throw error(oss.str());
  }
  return constraints_;
}

const OpIds &OpeningResult::schedule() const {
  if (status() != OpeningStatus::Valid) {
    std::ostringstream oss;
    oss << "Call to OpeningResults::schedule, on " << *this
        << ". This method is only valid for OpeningResults with "
        << "a changed schedule, non-changed schedules are not stored. ";
    throw error(oss.str());
  }
  return schedule_;
}

OpeningResult OpeningResult::validWithUnchangedSchedule(Constraints &&cs) {
  return OpeningResult(OpeningStatus::Valid, std::move(cs), {}, false);
}
OpeningResult OpeningResult::validWithChangedSchedule(Constraints &&cs,
                                                      OpIds &&sched) {
  return OpeningResult(
      OpeningStatus::Valid, std::move(cs), std::move(sched), true);
}

OpeningResult OpeningResult::cycle() {
  return OpeningResult(OpeningStatus::Cycle, {}, {}, false);
}

OpeningResult OpeningResult::alreadyOpen() {
  return OpeningResult(OpeningStatus::AlreadyOpen, {}, {}, false);
}

OpeningResult OpeningResult::notParallelWriteable() {
  return OpeningResult(OpeningStatus::NotParallelWriteable, {}, {}, false);
}

std::ostream &operator<<(std::ostream &ost, const OpeningStatuses &statuses) {
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
