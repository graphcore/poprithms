// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <tuple>

#include <poprithms/error/error.hpp>
#include <poprithms/logging/logging.hpp>
#include <testutil/schedule/shift/bifurcate_generator.hpp>
#include <testutil/schedule/shift/branch_doubling_generator.hpp>
#include <testutil/schedule/shift/diamond_generator.hpp>
#include <testutil/schedule/shift/grid_generator.hpp>
#include <testutil/schedule/shift/randomgraph.hpp>
#include <testutil/schedule/shift/recompute_generator.hpp>

namespace {

using Map = std::map<std::string, std::string>;
using namespace poprithms::schedule::shift;

auto getTestSuite() {

  // tieBreaker       allTCO      filterSusceptible
  // ----------------------------------------------
  std::vector<std::array<std::string, 3>> testSuiteStrings{
      {"FIFO", "1", "0"},
      {"FIFO", "0", "0"},
      {"RANDOM", "0", "1"},
      {"RANDOM", "0", "0"},
      {"GREEDY", "0", "0"}};

  std::vector<Map> testSuite;
  for (auto x : testSuiteStrings) {
    testSuite.push_back({{"tieBreaker", std::get<0>(x)},
                         {"allTCO", std::get<1>(x)},
                         {"filterSusceptible", std::get<2>(x)}});
  }
  return testSuite;
}

std::string mapstring(const Map &m) {
  std::ostringstream oss;
  oss << "[ ";
  for (auto x : m) {
    oss << x.first << ':' << x.second << ' ';
  }
  oss << ']';
  return oss.str();
}

class Logger {

  // The Graphs used for regression will be progressively "larger" until thet
  // are so large that the time to schedule them exceeds the time limit
  // (timeLimit()). The definition of "large" depends on the Graph type, and
  // managed by increaseCurrentSize()

public:
  virtual std::string getDescription() const                    = 0;
  virtual Graph getCurrent() const                              = 0;
  virtual void assertCorrectness(const ScheduledGraph &g) const = 0;
  virtual double timeLimit() const { return 6.0; }

  std::string getLogString() {
    std::cout << "\n>>> Processing Graph Type " << getDescription() << '.'
              << std::endl;
    std::ostringstream oss;
    for (auto params : getTestSuite()) {

      std::cout << "\n\nprocessing with scheduler parameters:\n "
                << mapstring(params) << std::endl;

      double deltaT{0};
      resetCurrentSize();
      while (deltaT < timeLimit()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        increaseCurrentSize();
        auto g  = getCurrent();
        auto sg = apply(params, g, oss);
        assertCorrectness(sg);
        auto t1 = std::chrono::high_resolution_clock::now();
        deltaT  = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "at " << g.nOps() << "     time taken was " << deltaT
                  << " [s]" << std::endl;
      }
    }

    resetCurrentSize();
    return oss.str();
  }

private:
  virtual void increaseCurrentSize() = 0;
  virtual void resetCurrentSize()    = 0;

  ScheduledGraph apply(const Map &params, const Graph &g, std::ostream &oss) {
    oss << "\n\ndescription=" << getDescription()
        << "\nnOpsBefore=" << g.nOps();
    auto t  = std::time(nullptr);
    auto tm = *std::localtime(&t);
    oss << "\nlogTime=" << std::put_time(&tm, "%d-%m-%Y at %H-%M");

    for (const auto &[k, v] : params) {
      oss << '\n' << k << '=' << v;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    auto sg = ScheduledGraph(Graph(g), params);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> tElapsed = t1 - t0;
    oss << std::scientific << std::setprecision(7);
    oss << '\n' << "timeTotal=" << tElapsed.count() << " [s]";
    oss << '\n' << "nOpsAfter=" << g.nOps();
    return sg;
  }
};

class BifurcateLogger : public Logger {
public:
  virtual std::string getDescription() const final { return "bifurcating"; }
  virtual Graph getCurrent() const final {
    return getBifurcatingGraph0(logN);
  }
  virtual void assertCorrectness(const ScheduledGraph &g) const final {
    assertGlobalMinimumBifurcatingGraph0(g, logN);
  }

private:
  uint64_t logN;
  virtual void increaseCurrentSize() final { ++logN; }
  virtual void resetCurrentSize() final { logN = 2; }
};

class RandomLogger : public Logger {
public:
  virtual std::string getDescription() const final { return "random"; }
  virtual Graph getCurrent() const final {
    return getRandomGraph(N + 100, E, D, graphSeed);
  }
  virtual void assertCorrectness(const ScheduledGraph &g) const final {}

private:
  virtual void increaseCurrentSize() final {
    N = static_cast<uint64_t>(N * 1.6);
  }
  virtual void resetCurrentSize() final { N = 100; }
  uint64_t N;

  static const constexpr uint64_t E         = 4;
  static const constexpr uint64_t D         = 15;
  static const constexpr uint64_t graphSeed = 1011;
};

class GridLogger : public Logger {
public:
  virtual std::string getDescription() const final { return "grid"; }
  virtual Graph getCurrent() const final { return getGridGraph0(nRows); }
  virtual void assertCorrectness(const ScheduledGraph &g) const final {
    assertGlobalMinimumGridGraph0(g, nRows);
  }

private:
  uint64_t nRows;
  virtual void increaseCurrentSize() final {
    nRows = static_cast<double>(nRows * 1.5);
  }
  virtual void resetCurrentSize() final { nRows = 5; }
};

class BranchDoublingLogger : public Logger {
public:
  BranchDoublingLogger(int _offset_) : offset(_offset_) {}
  virtual std::string getDescription() const final {
    return "branch-doubling";
  }
  virtual Graph getCurrent() const final {
    return getBranchDoublingGraph(nBranches, offset);
  }
  virtual void assertCorrectness(const ScheduledGraph &g) const final {
    assertGlobalMinimumBranchDoubling(g, nBranches, offset);
  }

private:
  int nBranches;
  int offset;

  virtual void increaseCurrentSize() final { ++nBranches; }
  virtual void resetCurrentSize() final { nBranches = 3; }
};

class DiamondLogger : public Logger {
public:
  virtual std::string getDescription() const final { return "diamond"; }
  virtual Graph getCurrent() const final { return getDiamondGraph0(N); }
  virtual void assertCorrectness(const ScheduledGraph &g) const final {
    assertGlobalMinimumDiamondGraph0(g, N);
  }

private:
  uint64_t N;
  virtual void increaseCurrentSize() final {
    N = static_cast<double>(N * 1.5) + 1;
  }
  virtual void resetCurrentSize() final { N = 5; }
};

class RecomputeLogger : public Logger {
public:
  virtual void assertCorrectness(const ScheduledGraph &g) const final {
    assertGlobalMinimumRecomputeGraph0(g);
  }

  virtual Graph getCurrent() const final {
    return getRecomputeGraph(getSeries(N));
  }

private:
  uint64_t N;
  virtual void increaseCurrentSize() final {
    N = static_cast<double>(N * 1.5) + 1;
  }
  virtual void resetCurrentSize() final { N = 20; }
  virtual std::vector<int> getSeries(uint64_t N) const = 0;
};

class LogRecomputeLogger : public RecomputeLogger {
public:
  virtual std::string getDescription() const final { return "log-recompute"; }

private:
  virtual std::vector<int> getSeries(uint64_t N) const {
    return getLogNSeries(N);
  }
};

class SqrtRecomputeLogger : public RecomputeLogger {
public:
  virtual std::string getDescription() const final {
    return "sqrt-recompute";
  }

private:
  virtual std::vector<int> getSeries(uint64_t N) const {
    return getSqrtSeries(N);
  }
};

} // namespace

int main(int argc, char **argv) {

  if (argc != 2) {
    std::ostringstream oss;
    oss << "\nWhile executing main of regression.cpp. "
        << "\nThe number of arguments received was argc=" << argc << '.'
        << "\nThe expected number of arguments was 1, "
        << " the name of the file to write logging information to. ";
    throw poprithms::test::error(oss.str());
  }

  std::ofstream out(argv[1]);

  out << DiamondLogger().getLogString();
  out << BifurcateLogger().getLogString();
  out << RandomLogger().getLogString();
  out << GridLogger().getLogString();
  out << LogRecomputeLogger().getLogString();
  out << SqrtRecomputeLogger().getLogString();
  out << BranchDoublingLogger(+1).getLogString();

  out.close();

  return 0;
}
