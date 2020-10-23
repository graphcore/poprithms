// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>

#include <poprithms/schedule/anneal/error.hpp>
#include <poprithms/schedule/anneal/graph.hpp>

namespace {

using namespace poprithms::schedule::anneal;

void test0() {

  // Cases where strings are not valid serialization strings:
  std::vector<std::string> testStrings;
  testStrings.push_back("");

  testStrings.push_back("\n");

  testStrings.push_back("\n\n\n\n");

  testStrings.push_back("{[{[{[");

  // addresses should increase contiguously
  testStrings.push_back(R"(
  {"ops":[
      {"address":6,"outs":[],"allocs":[],"debugString":"op0","fwdLink":1}], 
    "allocs":[]})");

  // order of appearance of keys matters
  testStrings.push_back(R"(
  {"ops":[
      {"outs":[],"address":6,"allocs":[],"debugString":"op0","fwdLink":1}], 
    "allocs":[]})");

  // we count how many strings are detected as invalid
  uint64_t badCatches{0};
  int i{0};
  for (auto x : testStrings) {
    try {
      auto g = Graph::fromSerializationString(x);
    } catch (const poprithms::error::error &e) {
      std::cout << "\nTest catch " << i << " : \n" << e.what() << std::endl;
      ++badCatches;
      ++i;
    }
  }

  if (badCatches != testStrings.size()) {
    throw error("Did not catch all bad serializations");
  }
}

void test1() {
  for (auto x : std::vector<char>{'\"'}) {
    Graph g;
    g.insertOp(std::string("ab") + std::string(1, x) + "cd");
    auto seriez = g.getSerializationString();
  }
}
} // namespace

int main() {
  test0();
  test1();
  return 0;
}
