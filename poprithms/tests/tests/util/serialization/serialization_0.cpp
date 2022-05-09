// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <memory>
#include <set>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <compute/host/serializer.hpp>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace {

void testComputeHostSerializationAlias0() {

  using namespace poprithms::compute::host;

  for (DType d : {DType::Float16,
                  DType::Float64,
                  DType::Int8,
                  DType::Int64,
                  DType::Unsigned8,
                  DType::Boolean}) {

    auto foo = Tensor::float32({6}, {0, 1, 0, 1, 0, 1}).to(d);

    // aliasing reverse.
    auto bar   = foo.reverse_(0);
    auto goo   = bar.slice_({1}, {4}).reverse_(0);
    auto who   = Tensor::float32({3}, {1, 0, 0}).to(d);
    auto endoo = Tensor::concat_({who, goo}, 0);

    // serialize foo and bar.
    std::vector<Tensor> ts{foo, bar, goo, foo, endoo};
    std::stringstream oss;
    {
      boost::archive::text_oarchive oa(oss);
      oa << ts;
    }

    // Reload foo and bar.
    Tensors reloaded;
    {
      boost::archive::text_iarchive ia(oss);
      ia >> reloaded;
    }

    // Assert that the loaded tensors are still aliased and have the correct
    // values.
    for (uint64_t i = 0; i < 3; ++i) {
      for (uint64_t j = 0; j < ts.size(); ++j) {
        reloaded.at(j).assertAllEquivalent(ts.at(j));
        reloaded.at(j).add_(1);
        ts.at(j).add_(1);
      }
    }
  }
}

void testComputeHostSerialization1() {

  using namespace poprithms::compute::host;

  auto foo = Tensor::int32(0).expand({16});
  auto bar = foo.reshape_({4, 4});

  auto size0 = [&]() -> int64_t {
    std::stringstream oss;
    boost::archive::text_oarchive oa(oss);
    oa << foo;
    return oss.str().size();
  }();

  auto size1 = [&]() -> int64_t {
    std::stringstream oss;
    boost::archive::text_oarchive oa(oss);
    oa << std::vector<Tensor>{foo, bar};
    return oss.str().size();
  }();

  if (size0 < 16) {
    throw poprithms::test::error("Impossible to store this string with 16 "
                                 "numbers with fewer than 16 characters");
  }

  if (size1 - size0 >= size0) {
    throw poprithms::test::error(
        "As the 2 tensors are aliased, the values "
        "should not be stored twice, should be a saving");
  }
}

void testComputeHostSerializationFromPointer() {

  using namespace poprithms::compute::host;

  std::vector<double> d{1, 2, 3};
  auto d_ = d.data();
  auto x  = Tensor::refFloat64({3}, d_);
  std::stringstream oss;

  bool caught{false};
  try {
    boost::archive::text_oarchive oa(oss);
    oa << x;

  } catch (const poprithms::error::error &) {
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error(
        "failed to intercept serialization of reference (PointerData)");
  }
}

void testComputeHostSerializationBool() {

  using namespace poprithms::compute::host;

  auto x = Tensor::boolean({3}, {true, false, false});
  std::stringstream oss;
  boost::archive::text_oarchive oa(oss);
  oa << x;

  boost::archive::text_iarchive ia(oss);
  auto y = Tensor::int16(17);
  ia >> y;

  y.assertAllEquivalent(x);
}

void testComputeHostSerializationHalf() {

  using namespace poprithms::compute::host;

  auto x0 = Tensor::float32({3}, {1.01, 2.2342, 1e-8});
  auto x  = x0.toFloat16();

  std::stringstream oss;
  boost::archive::text_oarchive oa(oss);
  oa << x;

  boost::archive::text_iarchive ia(oss);
  auto y = Tensor::int16(17);
  ia >> y;

  y.assertAllEquivalent(x);
}

void testTwoFloat16() {

  // In my initial implementation I had a cusmtom serialization of
  // vector<float16>, which would not work (I don't know why) when there were
  // multiple such vectors being serialized. The second one would not write
  // any data when storing. The current implementation rather serializes each
  // float16 itself, and leaves the vector specific work to the vector class.
  //
  // This test failed with the initial implementation.

  using namespace poprithms::compute::host;
  auto x0 = Tensor::uniformFloat16(-1, +1, {4}, 1011);
  auto x1 = x0.add(1);
  Tensors ts{x0, x1};

  std::stringstream oss;
  boost::archive::text_oarchive oa(oss);
  oa << ts;
  boost::archive::text_iarchive ia(oss);
  Tensors loaded;
  ia >> loaded;

  for (uint64_t i = 0; i < ts.size(); ++i) {
    ts.at(i).assertAllEquivalent(loaded.at(i));
  }
}

void testMulipleAtOnce() {

  using namespace poprithms::compute::host;
  auto x0  = Tensor::uniformFloat16(10, 10, {4}, 1011);
  auto x1  = Tensor::uniformFloat64(-1, 1, {5}, 1011).slice_({1}, {3});
  auto x2  = x1.reverse_(0);
  auto x3  = Tensor::concat_({x1, x2}, 0);
  auto x4  = Tensor::uniformFloat16(11, 11, {27}, 1011);
  auto x5  = Tensor::arangeInt8(0, 20, 2);
  auto x6  = x5.slice_({1}, {8});
  auto x7  = x6.slice_({1}, {3});
  auto x8  = Tensor::boolean({2}, {false, true}).reverse_(0).reverse_(0);
  auto x9  = x0.add(1.);
  auto x10 = x8.mul(0);
  auto x11 = x0.reverse_(0);

  std::vector<Tensor> all{x0, x4, x1, x2, x3, x5, x6, x7, x8, x9, x10, x11};
  std::stringstream oss;
  {
    boost::archive::text_oarchive oa(oss);
    oa << all;
  }

  Tensors loaded;
  {
    boost::archive::text_iarchive ia(oss);
    ia >> loaded;
  }

  for (uint64_t iter = 0; iter < 2; ++iter) {
    for (uint64_t i = 0; i < all.size(); ++i) {

      // check that the serialized aliasing is correct by inplace adding 1
      // to pre- and post- serialized tensors and checking that the side
      // effects on the other tensors are the same.
      all.at(i).assertAllEquivalent(loaded.at(i));
      if (!all.at(i).containsAliases()) {
        all.at(i).add_(1);
        loaded.at(i).add_(1);
      }
    }
  }
}

void testTensorShapes() {

  using namespace poprithms::ndarray;
  using namespace poprithms::compute::host;

  auto x = Tensor::zeros(DType::Int64, Shape({1, 2, 3, 4}));
  std::stringstream oss;
  {
    boost::archive::text_oarchive oa(oss);
    oa << x;
  }

  Tensor loaded = Tensor::ones(DType::Int16, {});
  {
    boost::archive::text_iarchive ia(oss);
    ia >> loaded;
  }

  if (loaded.shape() != Shape{1, 2, 3, 4} || loaded.dtype() != DType::Int64) {
    throw poprithms::test::error("Shape or type incorrectly loaded");
  }
}

} // namespace

int main() {
  testTensorShapes();
  testComputeHostSerializationAlias0();
  testComputeHostSerialization1();
  testComputeHostSerializationBool();
  testComputeHostSerializationHalf();
  testComputeHostSerializationFromPointer();
  testTwoFloat16();
  testMulipleAtOnce();
}
