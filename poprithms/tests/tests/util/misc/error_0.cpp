// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <iostream>
#include <ostream>
#include <sstream>
#include <vector>

#include <poprithms/error/error.hpp>

namespace {

void throw101() {
  throw poprithms::error::error(
      "testing", poprithms::error::Code(101), "Flat tyre.");
}

} // namespace

int main() {

  bool caught{false};

  // Throw an error, the 'what' message looks like :
  //  `poprithms::testing error, code is POPRITHMS101. Flat tyre.'
  try {
    throw101();
  } catch (const poprithms::error::error &e) {

    // Expect the code to be 101:
    if (e.code() == poprithms::error::Code(101ull)) {
      std::cout << e.what() << std::endl;
      caught = true;
    }
  }

  if (!caught) {
    throw poprithms::test::error("Failed to catch the error 101.");
  }
  return 0;
}
