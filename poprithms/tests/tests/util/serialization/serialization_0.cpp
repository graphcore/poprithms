// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <ndarray/serializer.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace {
void testShapeSerialization0() {

  using namespace poprithms::ndarray;

  Shape foo({1, 2, 3});
  std::stringstream oss;
  boost::archive::text_oarchive oa(oss);
  oa << foo;

  boost::archive::text_iarchive ia(oss);
  Shape bar({1});
  ia >> bar;

  if (foo != bar) {
    throw poprithms::test::error("Shape serialization failed");
  }
}
} // namespace

int main() { testShapeSerialization0(); }
