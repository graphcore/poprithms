#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <testutil/schedule/anneal/commandlineoptions.hpp>
#include <tuple>
#include <vector>
#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>
#include <poprithms/schedule/anneal/opalloc.hpp>

// recomputate graphs
//
// example of log-mem graph.
// N = 11
//
//  finish
//    b - b - b < b - b - b - b - b - b - b - b
//    ^   |   |   |   |   |   ^   |   |   |   ^
//    |   |   |   |   |   |   |   |   |   |   |
//    |   x   | / x   |   |   x   | / x   |   |
//    | / x - x - x - x   | / x - x - x - x   |
//    x > x - x - x - x - x - x - x - x - x > x
//  start
//
// n-times computed in forwards section :
//    1   3   2   3   2   1   3   2   3   2   1
//
// see recomp_illustration for a matplotlib generated pdf of the above "plot"
//
//
// example of sqrt-mem graph
// N = 9
//
//
// finish
//   b - b - b < b - b < b - b - b < b
//   ^   |   |   |   |   |   ^   |   |
//   |   |   |   |   |   |   |   |   |
//   |   x   x   x   |   x   x   x   |
//   x > x - x - x - x - x - x - x - x
// start
//
// n-times computed in forwards section :
//   1   2   2   2   1   2   2   2   1
//

poprithms::schedule::anneal::Graph
getRecomputeGraph(const std::vector<int> &nTimes) {
  using namespace poprithms::schedule::anneal;
  uint64_t N = nTimes.size();

  assert(!nTimes.empty());
  assert(nTimes.back() == 1);
  assert(nTimes[0] == 1);

  // assert that decreases are only by 1
  for (auto iter = std::next(nTimes.begin(), 1); iter != nTimes.end();
       ++iter) {
    auto prevIter = std::prev(iter, 1);
    if (*prevIter > *iter) {
      assert(*iter == *prevIter - 1);
    }
  }

  auto getFwdName = [](uint64_t layerIndex, uint64_t recompNumber) {
    return std::to_string(layerIndex) + '_' + std::to_string(recompNumber);
  };

  auto getBwdName = [](uint64_t layerIndex) {
    return "bwd_" + std::to_string(layerIndex);
  };

  Graph g;

  std::vector<std::vector<OpAlloc>> opAllocs;
  opAllocs.reserve(N);

  // forwards, forwards:
  for (uint64_t layerIndex = 0; layerIndex < nTimes.size(); ++layerIndex) {
    auto timesToRecomp = nTimes[layerIndex];
    opAllocs.push_back({});
    auto &layer = opAllocs.back();
    // will have 1 for each of the recomputations, and 1 for the backwards
    layer.reserve(timesToRecomp + 1);

    for (uint64_t n = 0; n < static_cast<uint64_t>(timesToRecomp); ++n) {
      auto mm = g.insertAlloc(1);
      std::vector<OpAddress> prods{};
      std::vector<AllocAddress> allocs{mm};
      if (layerIndex > 0) {
        const auto &prevLayer = opAllocs[layerIndex - 1];
        uint64_t prevLayerIndex =
            std::min<uint64_t>(prevLayer.size() - 1UL, n);
        prods.push_back(prevLayer[prevLayerIndex].op);
        allocs.push_back(prevLayer[prevLayerIndex].alloc);
      }
      auto op = g.insertOp(prods, allocs, getFwdName(layerIndex, n));
      layer.push_back({op, mm});
    }
  }

  // backwards, backwards:
  for (auto layer = opAllocs.rbegin(); layer != opAllocs.rend(); ++layer) {
    auto mm = g.insertAlloc(1);
    std::vector<OpAddress> prods{layer->back().op};
    std::vector<AllocAddress> allocs{mm, layer->back().alloc};
    if (layer != opAllocs.rbegin()) {
      const auto &prevLayer = *std::prev(layer, 1);
      prods.push_back(prevLayer.back().op);
      allocs.push_back(prevLayer.back().alloc);
    }
    auto op = g.insertOp(
        prods, allocs, getBwdName(std::distance(layer, opAllocs.rend()) - 1));
    layer->push_back({op, mm});
  }

  return g;
}

auto getLogNSeries(uint64_t N) {

  assert(N > 1);

  std::vector<int> series(N, 0);
  std::vector<bool> isSet(N, false);

  uint64_t nSet = 0;
  auto setTo    = [&nSet, &series, &isSet](uint64_t index, uint64_t value) {
    if (!isSet[index]) {
      series[index] = value;
      isSet[index]  = true;
      ++nSet;
    }
  };

  setTo(0, 1);
  setTo(series.size() - 1, 1);
  setTo((series.size() - 1) / 2, 1);

  int currentDepth = 2;
  while (nSet != N) {
    std::vector<uint64_t> setToUnset;
    std::vector<uint64_t> unsetToSet;
    for (uint64_t i = 0; i < N - 1; ++i) {
      if (isSet[i] == true && isSet[i + 1] == false) {
        setToUnset.push_back(i);
      }
      if (isSet[i] == false && isSet[i + 1] == true) {
        unsetToSet.push_back(i);
      }
    }

    assert(setToUnset.size() == unsetToSet.size());

    for (uint64_t i = 0; i < setToUnset.size(); ++i) {
      auto sTu = setToUnset[i];
      auto uTs = unsetToSet[i];
      assert(sTu < uTs);

      // uTs  sTu+1+(uTs-sTu)/2
      setTo(uTs, currentDepth);
      setTo(sTu + 1 + (uTs - sTu) / 2, currentDepth);
    }

    ++currentDepth;
    assert(nSet <= N);
  }

  return series;
}

int main(int argc, char **argv) {

  using namespace poprithms;
  using namespace poprithms::schedule::anneal;
  auto opts = CommandLineOptions::getCommandLineOptionsMap(
      argc,
      argv,
      {"N", "type"},
      {"The number of forward Ops",
       "The type of recomputation. Either sqrt: checkpoints at "
       "approximately every root(N) interval, or log: multi-depth "
       "recursion, where at each depth just the mid-point is checkpoint, "
       "and there approximately log(N) depths "});
  uint64_t nFwd             = std::stoi(opts.at("N"));
  std::string recomputeType = opts.at("type");
  std::vector<int> pattern;
  if (recomputeType == "sqrt") {
    uint64_t root = static_cast<uint64_t>(std::sqrt(nFwd));
    std::vector<int> sqrtPattern(nFwd, 2);
    sqrtPattern[0]     = 1;
    sqrtPattern.back() = 1;
    uint64_t c         = 0;
    while (root / 2 + c * root < nFwd) {
      sqrtPattern[root / 2 + c * root] = 1;
      c += 1;
    }
    pattern = sqrtPattern;
  } else if (recomputeType == "log") {
    pattern = getLogNSeries(nFwd);
  } else {
    throw poprithms::schedule::anneal::error(
        "Invalid type, log and sqrt are the current options");
  }
  auto g = getRecomputeGraph(pattern);
  g.initialize();
  std::cout << g.getLivenessString() << std::endl;

  g.minSumLivenessAnneal(
      CommandLineOptions::getAnnealCommandLineOptionsMap(opts));

  std::cout << g.getLivenessString() << std::endl;

  auto split = [](const std::string &x) {
    // auto found = x.find('_', 1);

    auto found = std::find(x.cbegin(), x.cend(), '_');
    if (*x.cbegin() == 'b') {
      return std::pair{std::stoi(std::string(std::next(found), x.cend())),
                       -1};
    } else {
      auto layer  = std::stoi(std::string(x.cbegin(), found));
      auto recomp = std::stoi(std::string(found + 1, x.cend()));
      return std::pair{layer, recomp};
    }
  };

  auto schedule = g.getScheduleToOp();
  std::vector<int> layers;
  layers.reserve(schedule.size());
  std::vector<int> recomps;
  recomps.reserve(schedule.size());
  for (auto x : schedule) {
    auto p = split(g.getOp(x).getDebugString());
    layers.push_back(p.first);
    recomps.push_back(p.second);
  }

  // some optimality tests:

  // get max layer
  auto maxLayer =
      std::accumulate(layers.cbegin(), layers.cend(), 0, [](int a, int b) {
        return std::max(a, b);
      });

  std::vector<std::vector<int>> recompOrder(
      static_cast<uint64_t>(maxLayer + 1));

  for (uint64_t i = 0; i < layers.size(); ++i) {
    int fwdCurrentRecomp = 0;
    // if it is bwds, it must be
    if (recomps[i] == -1) {
      // either preceded by bwd one before,
      if (recomps[i - 1] == -1 && layers[i - 1] - 1 == layers[i]) {
      }
      // or preceded by a fwd of the same layer.
      else if (recomps[i - 1] != -1 && layers[i - 1] == layers[i]) {
      } else {
        throw error("Bwd op in recompute test is not optimally scheduled");
      }
    }
    recompOrder[layers[i]].push_back(recomps[i]);
  }
  for (auto &x : recompOrder) {
    if (x.back() != -1) {
      throw error("expected final appearance of layer to be a gradient");
    }
    x.pop_back();
    for (auto iter = std::next(x.cbegin()); iter != x.cend(); ++iter) {
      if (*std::prev(iter, 1) >= *iter) {
        throw error("expected recomputation order to increase");
      }
    }
  }

  return 0;
}
