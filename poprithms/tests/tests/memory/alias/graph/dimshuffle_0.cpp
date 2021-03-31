// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <numeric>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>

int main() {
  using namespace poprithms::memory::alias;
  using namespace poprithms::util;

  // (1,2,3,4,5,6,7)
  std::vector<int64_t> sh(7, 0);
  std::iota(sh.begin(), sh.end(), 1);
  Shape shape(sh);

  // (1,2,3,4,5,6,0)
  std::vector<uint64_t> perm(7, 0);
  std::iota(perm.begin(), perm.end(), 1);
  perm.back() = 0;
  Permutation permutation(perm);

  Graph g;
  auto id0 = g.allocate(shape);

  // permute the Tensor 5*7 + offset times.
  // if offset is 0, expect the Tensor to have returned to its original state.
  for (int64_t offset : {0, 1}) {
    auto id = id0;
    for (int64_t i = 0; i < static_cast<int64_t>(sh.size()) * 5 + offset;
         ++i) {
      id = g.dimShuffle(id, permutation);
    }
    if (offset == 0 && g.tensor(id).shape() != g.tensor(id0).shape()) {
      std::ostringstream oss;
      oss << "The permutation should repeat every shape.rank() = "
          << shape.rank_u64() << " iterations. The number of iterations "
          << "(modulo " << shape.rank_u64() << ") is " << offset << '.';
      throw error(oss.str());
    }

    if (offset != 0 && g.tensor(id).shape() == g.tensor(id0).shape()) {
    }
  }

  return 0;
}
