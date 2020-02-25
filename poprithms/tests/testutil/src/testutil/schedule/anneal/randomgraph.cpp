#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <testutil/schedule/anneal/randomgraph.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

poprithms::schedule::anneal::Graph
getRandomGraph(uint64_t N, uint64_t E, uint64_t D, int graphSeed) {

  std::mt19937 gen(graphSeed);
  std::uniform_int_distribution<> distSizeAlloc(10, 19);

  std::vector<int> dBack(D);
  // -D ... -1
  std::iota(dBack.begin(), dBack.end(), -D);

  Graph g;

  for (int n = 0; n < N; ++n) {
    auto allocId = g.insertAlloc(distSizeAlloc(gen));
  }

  for (int n = 0; n < N; ++n) {
    auto n_u64 = static_cast<uint64_t>(n);
    if (n < D) {
      g.insertOp({}, {n_u64}, "op_" + std::to_string(n));
    } else {
      std::vector<int> samples;
      samples.reserve(E);
      std::sample(
          dBack.begin(), dBack.end(), std::back_inserter(samples), E, gen);
      for (auto &x : samples) {
        x += n;
      }
      std::vector<OpAddress> prods;
      std::vector<AllocAddress> allocs{n_u64};
      for (auto x : samples) {
        auto x_u64 = static_cast<uint64_t>(x);
        prods.push_back(x_u64);
        allocs.push_back(x_u64);
      }
      g.insertOp(prods, allocs, "op_" + std::to_string(n));
    }
  }
  return g;
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
