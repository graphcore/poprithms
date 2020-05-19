// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/error.hpp>
#include <poprithms/logging/logging.hpp>

namespace {
void fail(int stage) {
  throw poprithms::logging::error("Failed at stage " + std::to_string(stage));
}
} // namespace

int main() {
  using namespace poprithms::logging;

  //  A     B     C     D
  //
  //  Off   --    --    --
  Logger A("a");
  if (A.getLevel() != Level::Off) {
    fail(0);
  }

  //  Off   Off   --    --
  Logger B("b");
  if (A.getLevel() != Level::Off || B.getLevel() != Level::Off) {
    fail(1);
  }

  //  Info  Info  --    --
  setGlobalLevel(Level::Info);
  if (A.getLevel() != Level::Info || B.getLevel() != Level::Info) {
    fail(2);
  }

  //  Info  Info  Info  --
  Logger C("c");
  if (A.getLevel() != Level::Info || B.getLevel() != Level::Info ||
      C.getLevel() != Level::Info) {
    fail(3);
  }

  //  Debug Debug Debug --
  setGlobalLevel(Level::Debug);
  if (A.getLevel() != Level::Debug || B.getLevel() != Level::Debug ||
      C.getLevel() != Level::Debug) {
    fail(4);
  }

  //  Debug Off   Debug --
  B.setLevel(Level::Off);
  if (A.getLevel() != Level::Debug || B.getLevel() != Level::Off ||
      C.getLevel() != Level::Debug) {
    fail(5);
  }

  //  Info  Info  Info  --
  setGlobalLevel(Level::Info);
  if (A.getLevel() != Level::Info || B.getLevel() != Level::Info ||
      C.getLevel() != Level::Info) {
    fail(5);
  }

  //  Info  Info  Info  Info
  Logger D("d");
  if (A.getLevel() != Level::Info || B.getLevel() != Level::Info ||
      C.getLevel() != Level::Info || D.getLevel() != Level::Info) {
    fail(6);
  }

  //  Off   Info  Info  Info
  A.setLevel(Level::Off);
  if (A.getLevel() != Level::Off || B.getLevel() != Level::Info ||
      C.getLevel() != Level::Info || D.getLevel() != Level::Info) {
    fail(7);
  }

  bool caught = false;
  try {
    // The id "a" has already been used
    Logger E("a");
  } catch (const poprithms::util::error &e) {
    caught = true;
  }
  if (!caught) {
    throw error(
        "Failed to catch case where logger's with identical Ids are created");
  }

  return 0;
}
