// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <unordered_set>

#include <poprithms/util/error.hpp>
#include <poprithms/util/typedinteger.hpp>

int main() {

  using namespace poprithms::util;

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

  return 0;
}
