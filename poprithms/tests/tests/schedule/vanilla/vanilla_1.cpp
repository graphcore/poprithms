// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::schedule::vanilla;

//
// Canonical test case:
//
//     0
//     |
//     v
//  +--+--+
//  |     |
//  1     2
//  |     |
//  3     4
//  |     |
//  +--+--+
//     |
//     5
//
template <typename T> Edges<T> getTestEdges() {
  Edges<T> edges{{1, 2}, {3}, {4}, {5}, {5}, {}};
  return edges;
}

template <class T>
std::ostream &operator<<(std::ostream &ost, const std::vector<T> &ts) {
  poprithms::util::append(ost, ts);
  return ost;
}

template <class T>
void assertDistance(const std::vector<T> &schedule,
                    T x0,
                    T x1,
                    int64_t expectedDistance) {
  auto found0 = std::find(schedule.cbegin(), schedule.cend(), x0);
  auto found1 = std::find(schedule.cbegin(), schedule.cend(), x1);
  int64_t a   = std::distance(found0, found1);

  if (a != expectedDistance) {
    std::ostringstream oss;
    oss << "Expected the distance from " << x0 << " to " << x1 << " to be "
        << expectedDistance << " in this test, but in " << schedule
        << " it is " << a << '.';
    throw poprithms::test::error(oss.str());
  }
}

template <class T>
void assertContiguous(const std::vector<T> &schedule,
                      const std::vector<T> &subSchedule) {
  if (subSchedule.empty()) {
    return;
  }

  auto found0 = std::find(schedule.cbegin(), schedule.cend(), subSchedule[0]);
  auto x0     = std::distance(schedule.cbegin(), found0);
  for (uint64_t i = 1; i < subSchedule.size(); ++i) {
    if (x0 + i >= schedule.size() || schedule[x0 + i] != subSchedule[i]) {
      std::ostringstream oss;
      oss << "The sub-schedule " << subSchedule << " is not contiguous in "
          << " schedule: " << schedule << '.';
      throw poprithms::test::error(oss.str());
    }
  }
}

template <class T>
void assertOrder(const std::vector<T> &schedule,
                 const std::vector<T> &x0,
                 const std::vector<T> &x1) {
  if (x0.empty() || x1.empty()) {
    return;
  }

  std::vector<T> opToSchedule(schedule.size());
  for (uint64_t i = 0; i < schedule.size(); ++i) {
    opToSchedule[schedule[i]] = i;
  }

  int64_t m0 = -1;
  for (auto x : x0) {
    m0 = std::max<int64_t>(opToSchedule[x], m0);
  }

  int64_t m1 = schedule.size();
  for (auto x : x1) {
    m1 = std::min<int64_t>(opToSchedule[x], m1);
  }

  if (m0 >= m1) {
    std::ostringstream oss;
    oss << "Failed to assert order. "
        << "For the schedule " << schedule << ", with x0 = " << x0
        << " and x1 = " << x1 << ", max over x0 is " << m0
        << " and min over x1 is " << m1 << '.';
    throw poprithms::test::error(oss.str());
  }
}

// Test that the test methods are correct:
void testTest() {

  bool caughtOrder{false};
  bool caughtContiguous{false};
  bool caughtDistance{false};
  try {
    assertOrder<int>({10, 11, 12, 13, 9}, {10, 11, 13}, {9, 12});
  } catch (const poprithms::error::error &) {
    caughtOrder = true;
  }

  try {
    assertContiguous<int>({10, 5, 6, 7, 4}, {5, 6, 4});
  }

  catch (const poprithms::error::error &) {
    caughtContiguous = true;
  }

  try {

    // distance should be 3
    auto incorrectDistance = 2;
    assertDistance<int>({10, 5, 6, 7, 4}, 5, 4, incorrectDistance);
  }

  catch (const poprithms::error::error &) {
    caughtDistance = true;
  }

  if (!caughtOrder) {
    throw poprithms::test::error("Testing for order doesn't work");
  }

  if (!caughtContiguous) {
    throw poprithms::test::error("Testing for contiguity doesn't work");
  }

  if (!caughtDistance) {
    throw poprithms::test::error("Testing for distance doesn't work");
  }
}

template <typename T> void test0() {

  auto edges = getTestEdges<T>();

  // for fifo, must have {1,2} before {3,4}.
  {
    auto fifoSchedule = Scheduler<T, double>::fifo(
        edges, {}, {}, ErrorIfCycle::Yes, VerifyEdges::Yes);
    assertOrder<T>(fifoSchedule, {1, 2}, {3, 4});
    assertOrder<T>(fifoSchedule, {0}, {1, 2});
    assertOrder<T>(fifoSchedule, {3, 4}, {5});
  }

  // for filo, must have [1,3] and [2,4] contiguous.
  {
    auto filoSchedule = Scheduler<T, double>::filo(
        edges, {}, {}, ErrorIfCycle::Yes, VerifyEdges::Yes);
    assertDistance<T>(filoSchedule, 1, 3, 1);
    assertDistance<T>(filoSchedule, 2, 4, 1);
  }

  // for filo, can control order of [1,3] and [2,4] by using priorities.
  {
    auto filoSchedule = Scheduler<T, double>::filo(
        edges, {{1, -100.}}, {}, ErrorIfCycle::Yes, VerifyEdges::Yes);
    assertContiguous<T>(filoSchedule, {0, 2, 4, 1, 3, 5});
  }
  {
    auto filoSchedule = Scheduler<T, double>::filo(
        edges, {{1, 100.}}, {}, ErrorIfCycle::Yes, VerifyEdges::Yes);
    assertContiguous<T>(filoSchedule, {0, 1, 3, 2, 4, 5});
  }
  {
    auto filoSchedule = Scheduler<T, double>::filo(
        edges, {{2, 100.}}, {}, ErrorIfCycle::Yes, VerifyEdges::Yes);
    assertContiguous<T>(filoSchedule, {0, 2, 4, 1, 3, 5});
  }

  // filo with links
  {
    auto filoSchedule = Scheduler<T, double>::filo(
        edges, {}, {{1, 4}}, ErrorIfCycle::Yes, VerifyEdges::Yes);
    assertContiguous<T>(filoSchedule, {0, 2, 1, 4, 3, 5});
  }
  {
    auto filoSchedule = Scheduler<T, double>::filo(
        edges, {}, {{2, 3}}, ErrorIfCycle::Yes, VerifyEdges::Yes);
    assertContiguous<T>(filoSchedule, {0, 1, 2, 3, 4, 5});
  }

  // test all combinations of (links, priorities, scheduler).
  {
    std::vector<std::vector<std::array<T, 2>>> links{{}, {{2, 3}}};

    std::vector<std::vector<std::tuple<T, double>>> pris{
        {}, {{1, 10.}}, {{1, -10.}}};

    for (auto l : links) {
      for (const auto &p : pris) {
        for (std::string t : {"filo", "fifo", "random"}) {
          std::vector<T> sch;
          if (t == "filo") {
            sch = Scheduler<T, double>::filo(
                edges, p, l, ErrorIfCycle::Yes, VerifyEdges::Yes);
          } else if (t == "fifo") {
            sch = Scheduler<T, double>::fifo(
                edges, p, l, ErrorIfCycle::Yes, VerifyEdges::Yes);
          } else if (t == "random") {
            sch = Scheduler<T, double>::random(
                edges, p, l, 1011, ErrorIfCycle::Yes, VerifyEdges::Yes);
          }

          if (l.empty()) {
            assertOrder(sch, {0}, {1, 2});
            assertOrder(sch, {1}, {3});
            assertOrder(sch, {2}, {4});
            assertOrder(sch, {3, 4}, {5});
            if (p.empty()) {
            } else if (p == std::vector<std::tuple<T, double>>{{1, 10.}}) {
              assertOrder(sch, {1}, {2});
            } else if (p == std::vector<std::tuple<T, double>>{{1, -10.}}) {
              assertOrder(sch, {2}, {1});
            }
          } else if (l.size() == 1 && l[0] == std::array<T, 2>{2, 3}) {
            assertContiguous(sch, {0, 1, 2, 3, 4, 5});
          } else {
            throw poprithms::test::error("Link case not handled");
          }
        }
      }
    }
  }
}

// Check that an error message contains a sub-string.
void assertContains(const std::string &message, const std::string &fragment) {
  if (message.find(fragment) == std::string::npos) {
    std::ostringstream oss;
    oss << "\n\nExpected the message\n\"\"\"\n"
        << message << "\n\"\"\"\nto contain\n\"\"\"\n"
        << fragment << "\n\"\"\"\nbut it does not. ";
    throw poprithms::test::error(oss.str());
  }
}

void testErrors() {

  //
  //     0
  //     |
  //  +--+--+
  //  |     |
  //  1     2
  //  |     |
  //  3     4
  //  |     |
  //  +--+--+
  //     |
  //     5
  //

  // A cycle created by too many links:
  {
    bool caught{false};
    try {
      auto edges = getTestEdges<int64_t>();
      Scheduler<int64_t, double>::filo(
          edges, {}, {{1, 4}, {4, 5}}, ErrorIfCycle::Yes, VerifyEdges::Yes);
    } catch (const poprithms::error::error &e) {
      caught              = true;
      std::string message = e.what();
      assertContains(message, "there is a cycle in the graph");
      assertContains(
          message,
          "With all links removed (ignored), 6 of the 6 nodes are scheduled");
    }
    if (!caught) {
      throw poprithms::test::error("Failed to catch cycle ");
    }
  }

  // A cycle created by too many edges:
  {
    bool caught{false};
    try {
      auto edges = getTestEdges<int64_t>();
      edges[3].push_back(0);
      Scheduler<int64_t, double>::fifo(
          edges, {}, {{1, 3}}, ErrorIfCycle::Yes, VerifyEdges::Yes);
    } catch (const poprithms::error::error &e) {
      caught              = true;
      std::string message = e.what();
      assertContains(message, "there is a cycle in the graph");
      assertContains(
          message,
          "With all links removed (ignored), 0 of the 6 nodes are scheduled");
    }
    if (!caught) {
      throw poprithms::test::error("Failed to catch cycle ");
    }
  }

  // An invalid edge
  {
    bool caught{false};
    try {
      auto edges = getTestEdges<int64_t>();
      edges[3].push_back(edges.size());
      Scheduler<int64_t, double>::fifo(
          edges, {}, {{1, 3}}, ErrorIfCycle::Yes, VerifyEdges::Yes);
    } catch (const poprithms::error::error &e) {
      const std::string message = e.what();
      caught                    = true;
      assertContains(message, "Invalid edge");
    }
    if (!caught) {
      throw poprithms::test::error("Failed to catch bad edge");
    }
  }
}

// 200 runs on random graphs. Check that all constraints and link
// constraints are satisfied for fifo, filo, random.
void randomSoak() {

  for (uint64_t run = 0; run < 200; ++run) {

    // Ensure that all combinations of with/without for links and priorities
    // are tried:
    auto seed       = 1011 + run;
    uint64_t nOps   = 10 + run % 20;
    uint64_t nEdges = [nOps]() {
      if (nOps % 3 == 0) {
        return nOps / 2;
      } else if (nOps % 3 == 1) {
        return nOps * 2;
      } else {
        return nOps;
      }
    }();
    uint64_t nLinks       = 0 + 5 * (run % 3 == 0);
    uint64_t nPrioritized = 0 + 6 * (run % 2 == 1);
    std::mt19937 rng(seed);
    std::vector<int64_t> validSchedule(nOps);
    std::iota(validSchedule.begin(), validSchedule.end(), 0);
    std::shuffle(validSchedule.begin(), validSchedule.end(), rng);
    std::vector<std::vector<int64_t>> edges(nOps);

    for (uint64_t i = 0; i < nEdges; ++i) {
      auto a = rng() % nOps;
      auto b = rng() % (nOps - 1);
      b      = b + (b == a);
      if (a > b) {
        std::swap(a, b);
      }
      edges[validSchedule[a]].push_back(validSchedule[b]);
    }

    std::vector<std::array<int64_t, 2>> links;
    for (uint64_t l = 0; l < nLinks; ++l) {
      auto a = rng() % (nOps - 1);
      links.push_back({validSchedule[a], validSchedule[a + 1]});
    }

    std::vector<std::tuple<int64_t, double>> pris;
    for (uint64_t p = 0; p < nPrioritized; ++p) {
      pris.push_back({rng() % nOps, -1. + 2. * (rng() % 1000) / 1000.});
    }

    auto getBaseErr = [&run, &nLinks, &nPrioritized, &seed, &nOps]() {
      std::ostringstream oss;
      oss << "Failure in run #" << run << ". With nLinks=" << nLinks
          << ", nPrioritized=" << nPrioritized << ", seed=" << seed
          << ", and nOps=" << nOps << '.' << ' ';
      return oss.str();
    };

    auto assertValid = [&edges, &links, &getBaseErr](
                           const std::vector<int64_t> &sched) {
      const auto N = sched.size();
      if (N != edges.size()) {
        throw poprithms::test::error(getBaseErr() + "Incomplete schedule.");
      }
      std::vector<uint64_t> opToSched(N);
      for (uint64_t i = 0; i < N; ++i) {
        opToSched[sched[i]] = i;
      }
      for (uint64_t start = 0; start < N; ++start) {
        for (auto end : edges[start]) {
          if (opToSched[end] <= opToSched[start]) {
            throw poprithms::test::error(
                getBaseErr() + "Not all topological constraints satisfied.");
          }
        }
      }
      for (auto l : links) {
        auto x0 = opToSched[std::get<0>(l)];
        auto x1 = opToSched[std::get<1>(l)];
        if (x1 != x0 + 1) {
          throw poprithms::test::error(getBaseErr() +
                                       "Not all link constraints satisfied.");
        }
      }
    };

    assertValid(Scheduler<int64_t, double>::filo(
        edges, pris, links, ErrorIfCycle::Yes, VerifyEdges::Yes));
    assertValid(Scheduler<int64_t, double>::random(
        edges, pris, links, 1011, ErrorIfCycle::Yes, VerifyEdges::Yes));
    assertValid(Scheduler<int64_t, double>::fifo(
        edges, pris, links, ErrorIfCycle::Yes, VerifyEdges::Yes));
  }
}

} // namespace

int main() {
  testTest();
  test0<int64_t>();
  test0<uint64_t>();
  testErrors();
  randomSoak();
  return 0;
}
