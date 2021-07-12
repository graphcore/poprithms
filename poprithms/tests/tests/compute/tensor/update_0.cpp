// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>
#include <sstream>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::compute::host;

void testUpdate0() {

  const auto toUpdate = Tensor::int32(0).expand({1, 2, 3, 4});

  toUpdate.update_(Tensor::int32(1));
  toUpdate.assertAllEquivalent(Tensor::int32(1).expand({1, 2, 3, 4}));

  toUpdate.update_(Tensor::int32(2).expand({4}));
  toUpdate.assertAllEquivalent(Tensor::int32(2).expand({1, 2, 3, 4}));

  toUpdate.update_(Tensor::int32(3).expand({3, 4}));
  toUpdate.assertAllEquivalent(Tensor::int32(3).expand({1, 2, 3, 4}));

  toUpdate.update_(Tensor::int32(4).expand({2, 1, 4}));
  toUpdate.assertAllEquivalent(Tensor::int32(4).expand({1, 2, 3, 4}));

  toUpdate.update_(Tensor::int32(5).expand({1, 2, 3, 4}));
  toUpdate.assertAllEquivalent(Tensor::int32(5).expand({1, 2, 3, 4}));

  auto testBadBroadcast = [toUpdate](const Shape &updaterShape) {
    bool caught{false};
    try {
      const auto updater = Tensor::int32(1).expand(updaterShape);
      toUpdate.update_(updater);
    } catch (const poprithms::error::error &e) {
      caught = true;
    }
    if (!caught) {
      std::ostringstream oss;
      oss << "Expected update_ to fail, where toUpdate has Shape "
          << toUpdate.shape() << ", and updater has Shape " << updaterShape
          << ". ";
      throw poprithms::test::error(oss.str());
    }
  };

  testBadBroadcast({2});
  testBadBroadcast({10, 1, 3, 4});
  testBadBroadcast({1, 2, 2, 4});
}

void testUpdatePart0() {

  const auto toUpdate = Tensor::int32(0).expand({2, 3, 4});
  const auto updater  = Tensor::int32(1).expand({2, 3, 2});

  toUpdate.updatePart_(updater, Dimensions({2}), {0});
  toUpdate.assertAllEquivalent(Tensor::concat({updater, updater.zeros()}, 2));

  toUpdate.updatePart_(updater, Dimensions({2}), {2});
  toUpdate.assertAllEquivalent(Tensor::concat({updater, updater}, 2));
}

void testUpdatePart1() {

  auto testBadUpdate = [](const Dimensions &dims,
                          const std::vector<uint64_t> &starts,
                          const Tensor &updater) {
    const auto toUpdate = Tensor::unsigned16(0).expand({3, 3, 3});
    bool caught{false};
    try {
      toUpdate.updatePart_(updater, dims, starts);
    } catch (const poprithms::error::error &e) {
      caught = true;
    }
    if (!caught) {
      std::ostringstream oss;
      oss << "Expected to fail in call to updatePart_ with"
          << "\ntoUpdate = " << toUpdate << "\nupdater = " << updater
          << "\ndims = " << dims << "\nstarts = ";
      poprithms::util::append(oss, starts);
      throw poprithms::test::error(oss.str());
    }
  };

  // updated[1] should be 3
  testBadUpdate(
      Dimensions({0, 2}), {0, 0}, Tensor::unsigned16(1).expand({2, 2, 2}));

  // Starts is not of same size as Dimensions
  testBadUpdate(
      Dimensions({0, 2}), {0, 0, 0}, Tensor::unsigned16(1).expand({2, 3, 2}));

  // Invalid dimension
  testBadUpdate(
      Dimensions({3}), {0}, Tensor::unsigned16(1).expand({3, 3, 3}));

  // Invalid start
  testBadUpdate(
      Dimensions({1}), {3}, Tensor::unsigned16(1).expand({3, 1, 3}));
}

void testUpdatePart2() {
  const auto toUpdate = Tensor::float16(0).expand({2, 3, 4});
  toUpdate.updatePart_(Tensor::float16(1.0).expand({1, 1, 1}),
                       Dimensions({0, 1, 2}),
                       {1, 1, 1});

  const auto expected = Tensor::float16(0).expand({2, 3, 4});
  auto x              = expected.slice_({1, 1, 1}, {2, 2, 2});
  x.add_(Tensor::float16(1.));

  expected.assertAllEquivalent(toUpdate);
}

void oneHotTests() {

  // Vanilla test:
  {
    auto t0 = Tensor::randomInt32(-100, 100, {2, 3}, 1011);
    t0.encodeOneHot_({2, 0});
    //  [[ 0 0 1 ]
    //   [ 1 0 0 ]]
    std::cout << t0 << std::endl;
    t0.assertAllEquivalent(Tensor::int32({2, 3}, {0, 0, 1, 1, 0, 0}));
  }

  // Test where the tensor being encoded is not an origin tensor.
  {
    Tensor::concat_({Tensor::randomInt32(-100, 100, {2, 1}, 1011),
                     Tensor::randomInt32(-100, 100, {2, 2}, 1011)},
                    1)
        .encodeOneHot_({2, 0})
        .assertAllEquivalent(Tensor::int32({2, 3}, {0, 0, 1, 1, 0, 0}));
  }

  // test of the potential error cases:
  {
    bool caught{false};
    try {
      Tensor::randomInt32(-100, 100, {5, 4, 3}, 1011)
          .encodeOneHot_({0, 1, 0, 0, 1});
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "can't encode 3-d Tensorm should have caught this");
    }
  }

  {
    bool caught{false};
    try {
      Tensor::randomInt32(-100, 100, {3, 4}, 1011)
          .encodeOneHot_({0, 1, 0, 0});
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "expected 3 indices, not 4, should have caught this");
    }
  }

  {
    bool caught{false};
    try {
      Tensor::randomInt32(-100, 100, {3, 4}, 1011).encodeOneHot_({0, 1, 4});
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "index 4 should have been caught (too large)");
    }
  }

  {
    bool caught{false};
    try {
      // Test where the tensor being encoded self-aliases. Currently not
      // implemented, although it could be.
      Tensor::float32(1.0)
          .expand_({2, 3})
          .encodeOneHot_({2, 0})
          .assertAllEquivalent(Tensor::int32({2, 3}, {0, 0, 1, 1, 0, 0}));
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "has the case of encoding a self-aliasing tensor been "
          "implemented?");
    }
  }
}

} // namespace

int main() {
  testUpdate0();
  testUpdatePart0();
  testUpdatePart1();
  testUpdatePart2();
  oneHotTests();
}
