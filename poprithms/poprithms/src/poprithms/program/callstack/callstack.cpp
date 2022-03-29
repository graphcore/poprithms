// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <set>

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/program/callstack/calleetensorid.hpp>
#include <poprithms/program/callstack/callstack.hpp>
#include <poprithms/program/callstack/copymap.hpp>
#include <poprithms/program/callstack/stacktensorid.hpp>
#include <poprithms/program/callstack/stackutil.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>

// Why do we have multiple headers using a single source file (single
// translation unit for classes in different headers)? Because I'm trying to
// avoid tiny .cpp files, for better clang/gcc compile times.

namespace poprithms {
namespace program {
namespace callstack {

void CalleeTensorId::append(std::ostream &ost) const {
  ost << tId().str() << ':' << calleeIndex();
}

void StackTensorId::append(std::ostream &ost) const {
  ost << tId().str() << ':';
  poprithms::util::append(ost, callStack());
}

std::map<TensorId, uint64_t>
StackUtil::getCounts(const StackTensorIds &stids) {
  std::map<TensorId, uint64_t> m;
  for (const auto &tid : stids) {
    auto found = m.find(tid.tId());
    if (found != m.cend()) {
      ++found->second;
    } else {
      m.insert({tid.tId(), 1});
    }
  }
  return m;
}

std::ostream &operator<<(std::ostream &ost, const CalleeTensorId &id) {
  id.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const StackTensorId &id) {
  id.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const CalleeTensorIds &ids) {
  poprithms::util::append(ost, ids);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const StackTensorIds &ids) {

  using namespace poprithms::util;

  std::vector<std::string> tIds;
  std::vector<std::string> stacks;

  tIds.reserve(ids.size());
  stacks.reserve(ids.size());

  for (const auto &id : ids) {
    tIds.push_back(id.tId().str());
    std::ostringstream oss;
    append(oss, id.callStack());
    stacks.push_back(oss.str());
  }

  std::vector<poprithms::util::StringColumn> cols;
  cols.push_back(StringColumn("TensorId", tIds));
  cols.push_back(StringColumn("Call stack", stacks));
  const auto finalString = poprithms::util::alignedColumns(cols);
  ost << finalString;
  return ost;
}

StackTensorIds StackUtil::inScope(const TensorIds &tIds,
                                  const CallStack &callStack) {
  StackTensorIds nodes;
  nodes.reserve(tIds.size());
  for (const auto &tId : tIds) {
    nodes.push_back(StackTensorId(tId, callStack));
  }
  return nodes;
}
std::set<TensorId> StackUtil::tensorIds(const StackTensorIds &ids) {
  std::set<TensorId> sIds;
  for (const auto &id : ids) {
    sIds.insert(id.tId());
  }
  return sIds;
}

void CallEvent::append(std::ostream &ost) const {
  ost << "caller=" << caller() << ",callee=" << callee().get_u64();
  if (index_u64() != 0) {
    ost << ",index=" << index_u64();
  }
}

std::ostream &operator<<(std::ostream &ost, const CallEvent &cse) {
  cse.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const CallStack &cse) {
  poprithms::util::append(ost, cse);
  return ost;
}

const std::vector<std::pair<CallEvent, OutIndex>> &
CopyOutMap::get(const TensorId &tId) const {
  auto found = m_.find(tId);
  if (found != m_.cend()) {
    return found->second;
  }
  return empty_;
}

const std::vector<std::pair<CallEvent, InIndex>> &
CopyInMap::get(const TensorId &tId) const {
  auto found = m_.find(tId);
  if (found != m_.cend()) {
    return found->second;
  }
  return empty_;
}

} // namespace callstack
} // namespace program
} // namespace poprithms
