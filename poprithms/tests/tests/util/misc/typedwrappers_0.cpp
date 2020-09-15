// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <unordered_set>

#include <poprithms/util/error.hpp>
#include <poprithms/util/typedinteger.hpp>
#include <poprithms/util/typedvector.hpp>

namespace {

using namespace poprithms::util;

void testTypedInteger() {

  TypedInteger<'A', int> a(1);
  TypedInteger<'B', int> b(2);
  TypedInteger<'A', uint64_t> c(3);
  TypedInteger<'A', int> d(4);

  static_assert(!std::is_same<decltype(a), decltype(b)>::value,
                "different char");

  static_assert(!std::is_same<decltype(a), decltype(c)>::value,
                "different int type");

  static_assert(std::is_same<decltype(a.get()), decltype(b.get())>::value,
                "both int");

  static_assert(std::is_same<decltype(a), decltype(d)>::value,
                "same char, same int");

  a += 3;
  if (a != d) {
    throw error("Failed in typedinteger test : operator +=");
  }

  std::unordered_set<decltype(a)> usesHash;
  usesHash.insert(a);
}

void testTypedVector() {

  using T0 = TypedVector<int, 'T', '0'>;
  if (T0{1, 2, 3, 4}.size() != 4) {
    throw error("Incorrect number of elements in T0{1,2,3,4}");
  }
  if (T0{5, 5}.size() != 2) {
    throw error("Incorrect number of elements in T0{5,5}");
  }
  if (T0(std::vector(10, 8)).size() != 10) {
    throw error("Incorrect number of elements in T0(std::vector(10,8))");
  }

  using T1 = TypedVector<int, 'T', '1'>;
  static_assert(!std::is_same<T0, T1>::value, "Different types");
  static_assert(std::is_same<T1, T1>::value, "Different types");
}
} // namespace

int main() {
  testTypedInteger();
  testTypedVector();
  return 0;
}
