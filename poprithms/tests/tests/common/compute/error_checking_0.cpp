// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>
#include <string>

#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/error/error.hpp>

void test0() {
  using namespace poprithms::common::compute;

  SlickGraph m;
  auto sg0 = m.createSubGraph("sg0");

  auto x0 = sg0.hostInt32Variable({5});

  // invalid shape : cannot be added to x0.
  auto y0 = sg0.hostInt32Variable({4});

  // invalid type : cannot be added to x0.
  auto y1 = sg0.hostFloat32Variable({5});

  // invalid device : cannot be added to x0.
  auto y2 = sg0.variable(DType::Int32, {5}, m.rootIpu());

  for (auto tBad : {y0, y1, y2}) {
    bool caught{false};
    try {
      x0 + tBad;
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      std::ostringstream oss;
      oss << "Failed to catch error when adding incompatible tensors. "
          << "The tensors have infos " << m.tensorInfo(x0) << " and "
          << m.tensorInfo(tBad) << ".";
      throw poprithms::test::error(oss.str());
    }
  }
}

int main() {
  test0();
  return 0;
}
