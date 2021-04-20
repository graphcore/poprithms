// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <stdexcept>
#include <string>

// Generarted header for this test, which contains all poprithms headers
#include "all_headers.hpp"

// From https://stackoverflow.com/questions/2324658/
//        how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
std::string getCxxVersionSting() {
  switch (__cplusplus) {
  case 199711L:
    return "98";
  case 201103L:
    return "11";
  case 201402L:
    return "14";
  case 201703L:
    return "17";
  default:
    return "unknown";
  }
}

int main() {
  // Verify that we have included the headers
  poprithms::schedule::shift::AllocAddress address = 4;
  (void)address;
  auto pm = poprithms::schedule::transitiveclosure::TransitiveClosure({});

  // Verify that this test has been compiled with the C++11 standard - this
  // should have been enforced via CMake
  std::string cxxVersion = getCxxVersionSting();
  bool isCxx11           = (cxxVersion.compare("11") == 0);
  if (!isCxx11) {
    throw std::runtime_error(
        "This test must be compiled with C++11, but you are using C++" +
        cxxVersion);
  }

  return 0;
}
