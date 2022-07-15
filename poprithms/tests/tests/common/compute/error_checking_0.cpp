// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <sstream>
#include <string>

#include <poprithms/common/compute/slickgraph.hpp>
#include <poprithms/error/error.hpp>

namespace {

using namespace poprithms::common::compute;
void testIncompatibleInputs0() {

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

void testDoubleInplace0() {

  SlickGraph m;
  auto sg0 = m.createSubGraph("sg0");
  auto x0  = sg0.hostFloat32Variable({3, 4, 5});
  auto x1  = x0.abs_();
  auto x2  = x0.sin_();

  (void)x1;
  (void)x2;

  bool caught{false};
  try {
    auto x = m.vanillaSchedule();
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Failed to catch error of double write");
  }
}

void testDoubleInplace1() {

  SlickGraph m;
  auto sg0 = m.createSubGraph("sg0");
  auto x0  = sg0.hostFloat32Variable({4});

  //  0 1 2 3
  //  =======
  //  a a . .
  //  . . s s
  //  . . r .
  //
  auto a = x0.slice_({0}, {2}).abs_();
  auto s = x0.slice_({2}, {4}).sin_();
  auto r = x0.slice_({2}, {3}).relu();
  m.constraint(a.opId(), r.opId());

  // fine: (a -> r -> s).
  m.vanillaSchedule();

  // can't make sin_ before relu
  {
    bool caught{false};
    try {
      m.constraint(s.opId(), r.opId());
      m.vanillaSchedule();
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error("relu cannot be after sin_");
    }
  }

  // can't have 2 inplace modifiers of element 1:
  {
    bool caught{false};
    try {
      x0.slice_({1}, {2}).cos_();
      m.vanillaSchedule();
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error("Slices overlap, double write");
    }
  }
}
} // namespace

int main() {
  testIncompatibleInputs0();
  testDoubleInplace0();
  testDoubleInplace1();
  return 0;
}
