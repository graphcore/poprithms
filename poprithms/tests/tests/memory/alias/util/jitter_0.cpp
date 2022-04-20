// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/alias/jitgrower.hpp>
#include <poprithms/memory/alias/mapper.hpp>

namespace {
using namespace poprithms::memory;

using TensorId  = int;
using TensorIds = std::vector<TensorId>;

class TestMapper final : public alias::Mapper<TensorId> {
public:
  std::string external() const final { return "TestMapper"; }
};

class JitTestGrower final : public alias::JitGrower<TensorId> {

private:
  std::map<TensorId, TensorIds> fwdEdges_;
  std::map<TensorId, TensorIds> bwdEdges_;
  TestMapper mapper_;
  alias::Graph aliasGraph;

  std::vector<TensorId> growHistory_;

public:
  const alias::Graph &graph() const { return aliasGraph; }

  const TestMapper &mapper() const { return mapper_; }

  JitTestGrower(const std::map<TensorId, TensorIds> &fwdEdges)
      : fwdEdges_(fwdEdges) {

    // build backwards edges:
    for (const auto &[f, ts] : fwdEdges_) {
      bwdEdges_.insert({f, {}});
    }
    for (const auto &[f, ts] : fwdEdges_) {
      for (auto t : ts) {
        if (bwdEdges_.find(t) == bwdEdges_.cend()) {
          bwdEdges_.insert({t, {}});
        }
      }
    }

    for (const auto &[f, ts] : fwdEdges_) {
      for (auto t : ts) {
        bwdEdges_.at(t).push_back(f);
      }
    }
    std::cout << "JitTestGrower created" << std::endl;
  }

  TensorIds aliasingIns(const TensorId &tId) const final {
    std::cout << "getting aliasing ins of " << tId << std::endl;
    return bwdEdges_.at(tId);
  }

  bool containsAliasTensor(const TensorId &tId) const final {
    return mapper_.has(tId);
  }

  // Toy model:
  // - inputs (allocations) are all shape {1},
  // - there are ops which perform reverse
  // - there are ops with perform concat.
  void growAliasTensors(const TensorIds &scheduled) final {
    for (auto x : scheduled) {
      std::cout << "going to in grow " << x << std::endl;
      auto aliId = [x, this]() {
        auto aliIns = mapper_.ids(bwdEdges_.at(x));
        if (aliIns.size() == 0) {
          return aliasGraph.allocate({1}, alias::Color(17));
        } else if (aliIns.size() == 1) {
          return aliasGraph.reverse(aliIns[0], {0});
        } else {
          return aliasGraph.concat(aliIns, 0);
        }
      }();

      growHistory_.push_back(x);

      mapper_.insert({aliId}, {x});
    }
  }

  const std::vector<TensorId> &growHistory() const { return growHistory_; }
};

std::ostream &operator<<(std::ostream &ost, const TensorIds &vs) {
  poprithms::util::append(ost, vs);
  return ost;
}

void assertHistory(
    const JitTestGrower &jt,
    TensorIds contains,
    const std::vector<std::pair<TensorId, TensorId>> &constraints) {

  const auto &history = jt.growHistory();
  auto sortedHistory  = history;
  std::sort(sortedHistory.begin(), sortedHistory.end());
  std::sort(contains.begin(), contains.end());
  if (sortedHistory != contains) {
    std::ostringstream oss;
    oss << "The set of grown tensors expected to be " << contains
        << ", but it is " << sortedHistory;
    throw poprithms::test::error(oss.str());
  }

  auto orderCorrect = [&history](auto p) {
    for (const auto x : history) {
      if (x == p.first) {
        return;
      }
      if (x == p.second) {
        std::ostringstream oss;
        oss << "Expected to observe " << p.first << " before " << p.second
            << " in the grow history, but did not. History is " << history;
        throw poprithms::test::error(oss.str());
      }
    }
  };

  for (const auto &p : constraints) {
    orderCorrect(p);
  }
}

void test0() {

  //  0 --> 1 --> 2 --> 3.
  JitTestGrower jTester({{0, {1}}, {1, {2}}, {2, {3}}});
  jTester.extend({1});
  assertHistory(jTester, {0, 1}, {{0, 1}});
  jTester.extend({0});
  assertHistory(jTester, {0, 1}, {{0, 1}});
  jTester.extend({3});
  assertHistory(jTester, {0, 1, 2, 3}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}});
}

void test1() {

  /*
   *
   *     +--1--+
   *     |     |
   * 0 --+--2--+--> 4
   *     |     |
   *     +--3--+
   *
   *  5 ----------> 6
   *
   * */
  JitTestGrower jTester(
      {{0, {1, 2, 3}}, {1, {4}}, {2, {4}}, {3, {4}}, {5, {6}}});

  jTester.extend({1, 3});
  assertHistory(jTester, {0, 1, 3}, {{0, 1}, {0, 3}});
  jTester.extend({6, 4});
  assertHistory(jTester,
                {0, 1, 2, 3, 4, 5, 6},
                {{0, 1}, {0, 3}, {0, 2}, {2, 4}, {1, 4}, {3, 5}, {5, 6}});

  if (!jTester.graph().areAliased(jTester.mapper().id(5),
                                  jTester.mapper().id(6))) {
    throw poprithms::test::error("5 and 6 are aliased.");
  }

  if (jTester.graph().areAliased(jTester.mapper().id(5),
                                 jTester.mapper().id(0))) {
    throw poprithms::test::error("5 and 0 are NOT aliased.");
  }

  if (!jTester.graph().areAliased(jTester.mapper().id(4),
                                  jTester.mapper().id(0))) {
    throw poprithms::test::error("4 and 0 are aliased.");
  }
}

} // namespace

int main() {
  test0();
  test1();
  return 0;
}
