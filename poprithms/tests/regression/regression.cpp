#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <testutil/schedule/anneal/bifurcate_generator.hpp>
#include <testutil/schedule/anneal/branch_doubling_generator.hpp>
#include <testutil/schedule/anneal/diamond_generator.hpp>
#include <testutil/schedule/anneal/grid_generator.hpp>
#include <testutil/schedule/anneal/randomgraph.hpp>
#include <testutil/schedule/anneal/recompute_generator.hpp>

namespace {

using Map = std::map<std::string, std::string>;
using namespace poprithms::schedule::anneal;

auto getTestSuite() {
  std::vector<std::tuple<Map, Map>> testSuite;

  testSuite.push_back({{{"tieBreaker", "RANDOM"}},
                       {{"logging", "0"},
                        {"pStayPut", "1.0"},
                        {"pHigherFallRate", "0.0"},
                        {"pClimb", "0.0"}}});

  testSuite.push_back({{{"tieBreaker", "FIFO"}},
                       {{"logging", "0"},
                        {"pStayPut", "1.0"},
                        {"pHigherFallRate", "0.0"},
                        {"pClimb", "0.0"}}});

  testSuite.push_back({{{"tieBreaker", "GREEDY"}},
                       {{"logging", "0"},
                        {"pStayPut", "1.0"},
                        {"pHigherFallRate", "0.0"},
                        {"pClimb", "0.0"}}});

  testSuite.push_back({{{"tieBreaker", "RANDOM"}},
                       {{"logging", "0"},
                        {"pStayPut", "50.0"},
                        {"pHigherFallRate", "0.0"},
                        {"pClimb", "0.5"}}});

  testSuite.push_back({{{"tieBreaker", "FIFO"}},
                       {{"logging", "0"},
                        {"pStayPut", "4.0"},
                        {"pHigherFallRate", "1.0"},
                        {"pClimb", "0.5"}}});
  return testSuite;
}

class Logger {

public:
  virtual std::string getDescription() const           = 0;
  virtual Graph getCurrent() const                     = 0;
  virtual void assertCorrectness(const Graph &g) const = 0;
  virtual double timeLimit() const { return 6.0; }

  std::string getLogString() {
    std::cout << "\n\n"
              << "Processing " << getDescription() << std::endl;
    std::ostringstream oss;
    for (auto iMap_aMap : getTestSuite()) {
      std::cout << "\n"
                << "Processing next settings" << std::endl;
      double deltaT{0.0};
      resetCurrentSize();
      while (deltaT < timeLimit()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        increaseCurrentSize();
        auto g = getCurrent();
        apply(std::get<0>(iMap_aMap), std::get<1>(iMap_aMap), g, oss);
        assertCorrectness(g);
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

  void apply(const Map &initializeMap,
             const Map &annealMap,
             Graph &g,
             std::ostream &oss) {
    oss << "\n\ndescription=" << getDescription()
        << "\nnOpsBefore=" << g.nOps();
    for (const auto &[k, v] : initializeMap) {
      oss << '\n' << k << '=' << v;
    }
    for (const auto &[k, v] : annealMap) {
      oss << '\n' << k << '=' << v;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    g.initialize(initializeMap);
    auto t1 = std::chrono::high_resolution_clock::now();
    g.minSumLivenessAnneal(annealMap);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> initElapsed   = t1 - t0;
    std::chrono::duration<double> annealElapsed = t2 - t1;
    oss << std::scientific << std::setprecision(7);
    oss << '\n' << "timeInitialize=" << initElapsed.count() << " [s]";
    oss << '\n' << "timeAnneal=" << annealElapsed.count() << " [s]";
    oss << '\n' << "nOpsAfter=" << g.nOps();
  }
};

class BifurcateLogger : public Logger {
public:
  virtual std::string getDescription() const final { return "bifurcating"; }
  virtual Graph getCurrent() const final {
    return getBifurcatingGraph0(logN);
  }
  virtual void assertCorrectness(const Graph &g) const final {
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
  virtual void assertCorrectness(const Graph &g) const final {}

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
  virtual void assertCorrectness(const Graph &g) const final {
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
  virtual void assertCorrectness(const Graph &g) const final {
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
  virtual void assertCorrectness(const Graph &g) const final {
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
  virtual void assertCorrectness(const Graph &g) const final {
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

int main() {

  std::ofstream out("logging_file_name.txt");

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
