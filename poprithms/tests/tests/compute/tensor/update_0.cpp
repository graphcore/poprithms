// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>
#include <sstream>

#include <poprithms/compute/host/error.hpp>
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
      throw error(oss.str());
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
      throw error(oss.str());
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

} // namespace

int main() {
  testUpdate0();
  testUpdatePart0();
  testUpdatePart1();
  testUpdatePart2();
}
