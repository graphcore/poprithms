// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/schedule/transitiveclosure/error.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

int main() {

  using namespace poprithms::schedule::transitiveclosure;
  /*
   *
   *     0
   *    /|\
   *   1 2 3
   *    \|/
   *     4
   *     |
   *     5
   *
   *  */
  TransitiveClosure em{{{1, 2, 3}, {4}, {4}, {4}, {5}, {}}};

  auto x0 = em.getExtremumStatuses({0, 2, 5});
  if (x0.size() != 3) {
    throw error("Expected output of getExtremumStatuses to 1-1 with input");
  }

  if (x0[0] != std::tuple{IsFirst::Yes, IsFinal::No}) {
    throw error("Expected \"0\" to be first and not final");
  }
  if (x0[1] != std::tuple{IsFirst::No, IsFinal::No}) {
    throw error("Expected \"1\" to be in middle");
  }
  if (x0[2] != std::tuple{IsFirst::No, IsFinal::Yes}) {
    throw error("Expected \"0\" to be final and not first");
  }

  x0 = em.getExtremumStatuses({3});
  if (x0[0] != std::tuple{IsFirst::Yes, IsFinal::Yes}) {
    throw error("Exect getExtremumStatuses, called on a singleton, to "
                "always return {Yes, Yes}");
  }

  x0 = em.getExtremumStatuses({1, 2, 3});
  for (const auto &x : x0) {
    if (x != std::tuple{IsFirst::Maybe, IsFinal::Maybe}) {
      throw error("Expected {Maybe, Maybe} for all diamond edges");
    }
  }

  x0 = em.getExtremumStatuses({2, 1});
  for (const auto &x : x0) {
    if (x != std::tuple{IsFirst::Maybe, IsFinal::Maybe}) {
      throw error("Expected {Maybe, Maybe} for all diamond edges");
    }
  }

  x0 = em.getExtremumStatuses({5, 1, 2, 3});
  for (uint64_t i = 1; i < 4; ++i) {
    if (x0[i] != std::tuple{IsFirst::Maybe, IsFinal::No}) {
      throw error("Expected {Maybe, No} for diamond edge with peak");
    }
  }

  return 0;
}
