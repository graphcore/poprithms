// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <common/multiout/error.hpp>

#include <poprithms/common/multiout/removalevent.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace common {
namespace multiout {

bool RemovalEvent::operator==(const RemovalEvent &re) const {
  return opId == re.opId && name == re.name &&
         totalOpsCreatedSoFar == re.totalOpsCreatedSoFar &&
         context == re.context;
}

RemovalEvent::RemovalEvent(OpId id,
                           const std::string &name_,
                           uint64_t createdSoFar,
                           const std::string &context_)
    : opId(id), name(name_), totalOpsCreatedSoFar(createdSoFar),
      context(context_) {}

void RemovalEvent::append(std::ostream &ost) const {
  ost << "OpId:" << opId << ", name:\"" << name
      << "\", totalOpsCreatedSoFar:" << totalOpsCreatedSoFar << ", context:\""
      << context << "\".";
}

std::ostream &operator<<(std::ostream &ost, const RemovalEvent &e) {
  e.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const RemovalEvents &e) {
  e.append(ost);
  return ost;
}

std::string RemovalEvent::str() const {
  std::ostringstream oss;
  append(oss);
  return oss.str();
}

bool RemovalEvents::registered(OpId id) const {
  for (const auto &e : events) {
    if (e.opId == id) {
      return true;
    }
  }
  return false;
}

RemovalEvent RemovalEvents::event(OpId id) const {
  for (const auto &e : events) {
    if (e.opId == id) {
      return e;
    }
  }
  throw error("No RemovalEvent with OpId " + std::to_string(id.get()) + ". ");
}

void RemovalEvents::append(std::ostream &oss) const { oss << str(); }

std::string RemovalEvents::str() const {
  using Strings      = std::vector<std::string>;
  const auto nEvents = events.size();
  Strings opId(nEvents, "");
  Strings creations(nEvents, "");
  Strings name(nEvents, "");
  Strings ctxt(nEvents, "");
  for (uint64_t i = 0; i < events.size(); ++i) {
    const auto &e = events[i];
    opId[i]       = std::to_string(e.opId.get());
    creations[i]  = std::to_string(e.totalOpsCreatedSoFar);
    name[i]       = e.name;
    ctxt[i]       = e.context;
  }

  std::vector<util::StringColumn> cols;
  cols.push_back({"op id", opId, {}});

  if (std::any_of(
          name.cbegin(), name.cend(), [](auto &&x) { return !x.empty(); })) {
    cols.push_back({"op name", name, {}});
  }

  cols.push_back({"n-ops created when removed", creations, {}});

  if (std::any_of(
          ctxt.cbegin(), ctxt.cend(), [](auto &&x) { return !x.empty(); })) {
    cols.push_back({"context", ctxt, {}});
  }

  return util::alignedColumns(cols);
}
} // namespace multiout
} // namespace common
} // namespace poprithms
