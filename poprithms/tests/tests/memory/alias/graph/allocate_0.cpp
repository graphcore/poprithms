// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

int main() {

  using namespace poprithms::memory::alias;

  Graph g;

  int64_t nAllocs = 10;

  std::vector<AllocId> ids;
  for (int64_t i = 0; i < nAllocs; ++i) {
    Shape s({i, i});
    auto id = g.allocate(s);
    if (g.shape(id) != s) {
      throw error("Failed in shape comparison (from Graph) in allocate_0");
    }

    if (g.tensor(id).shape() != s) {
      throw error("Failed in shape comparison (from Tensor) in allocate_0");
    }
  }
  return 0;
}
