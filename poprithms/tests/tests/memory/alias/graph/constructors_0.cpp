// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <numeric>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/alias/mapper.hpp>

namespace {
using namespace poprithms::memory::alias;
using namespace poprithms::util;

void testMapper() {

  class StringMapper : public Mapper<std::string> {
    std::string external() const final { return "Stringy"; }
  };

  StringMapper mapper;
  auto foo = mapper.graph().allocate({2}, Color(0));
  auto bar = mapper.graph().allocate({2}, Color(1));
  mapper.insert({foo, bar}, {"foo", "bar"});
  auto bla = mapper.tensors({"foo", "foo", "bar"});

  if (mapper.id("foo") != foo) {
    throw poprithms::test::error("Failed to retrieve alias TensorId foo for "
                                 "Stringy tensor id \"foo\".");
  }

  if (mapper.ids({"bar", "foo"}) != TensorIds{bar, foo}) {
    throw poprithms::test::error(
        "Failed to retrieve alias TensorIds (foo,bar) for "
        "Stringy tensor ids (\"foo\",\"bar\").");
  }

  if (mapper.idFromAliasId(foo) != "foo") {
    throw poprithms::test::error("Failed to retrieve Stringy tensor id "
                                 "\"foo\" for alias TensorId foo");
  }
  if (mapper.idsFromAliasIds({foo, bar}) !=
      std::vector<std::string>{"foo", "bar"}) {
    throw poprithms::test::error(
        "Failed to retrieve Stringy tensors id (\"foo\",\"bar\") for alias "
        "TensorIds (foo,bar)");
  }

  if (mapper.tensor("foo").id() != foo) {
    throw poprithms::test::error("Failed in tensor(ExternTensorId)");
  }
  if (Graph::ids(mapper.tensors({"foo", "bar"})) != TensorIds{foo, bar}) {
    throw poprithms::test::error("Failed in tensors(ExternTensorIds)");
  }

  if (mapper.tensorFromAliasId(foo).id() != foo) {
    throw poprithms::test::error("Failed in tensorFromAliasId");
  }
  if (Graph::ids(mapper.tensorsFromAliasIds({foo, bar})) !=
      TensorIds{foo, bar}) {
    throw poprithms::test::error("Failed in tensorsFromAliasIds");
  }

  bool caught0{false};
  try {
    mapper.insert({foo}, {"another foo"});
  } catch (const poprithms::error::error &e) {
    caught0       = true;
    std::string w = e.what();
    if (w.find("Stringy") == std::string::npos) {
      throw poprithms::test::error(
          "Failed to find custom name in error string");
    }
  }
  if (!caught0) {
    throw poprithms::test::error("Failed to catch bad insert");
  }
}

void testConstrutors0() {

  Graph g;

  // If you know how many Tensors the Graph will have, this reserves the
  // required memory in the relevant vectors:
  g.reserve(22);

  auto arr0 = g.allocate({200});
  auto arr1 = g.concat({arr0, arr0, arr0}, 0);
  auto arr2 = g.reshape({arr1}, {100, 6});
  auto arr3 = g.dimShuffle({arr2}, {{1, 0}});
  g.reverse(arr3, {0});
  g.allocate({1, 2, 3});
  auto ali2 = g.allAliases(arr2);

  // Test the "rule-of-5"
  auto g2 = g;
  g       = g2;

  auto g3 = g;
  auto g5 = g;

  auto g4 = std::move(g);
  g5      = std::move(g3);

  //   id  type           ins      shape    outs  aliases  aliased to
  //   --- -------------- -------- -------- ----- -------- ------------
  //   0   Allocate       ()       (200)    (1)   no       (0,1,2,3,4)
  //   1   Concat         (0,0,0)  (600)    (2)   yes      (0,1,2,3,4)
  //   2   Reshape        (1)      (100,6)  (3)   yes      (0,1,2,3,4)
  //   3   Permute (1,0)  (2)      (6,100)  (4)   yes      (0,1,2,3,4)
  //   4   Reverse (0)    (3)      (6,100)  ()    yes      (0,1,2,3,4)
  //   5   Allocate       ()       (1,2,3)  ()    no       (5)
  std::cout << g5 << std::endl;

  if (g5 != g2) {
    throw poprithms::test::error(
        "Failed Graph comparison in constuctors test");
  }

  // identical, except dimshuffle switched
  Graph g6;
  arr0 = g6.allocate({200});
  arr1 = g6.concat({arr0, arr0, arr0}, 0);
  arr2 = g6.reshape({arr1}, {100, 6});
  arr3 = g6.dimShuffle({arr2}, {{0, 1}});
  g6.reverse(arr3, {0});
  g6.allocate({1, 2, 3});
  if (g6 == g2) {
    throw poprithms::test::error("g6 is different, failed in comparison");
  }
}
} // namespace

int main() {

  testMapper();
  testConstrutors0();

  return 0;
}
