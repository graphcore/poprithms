// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <thread>

#include <poprithms/logging/error.hpp>
#include <poprithms/logging/timeinscopeslogger.hpp>

namespace {

using namespace poprithms::logging;

void summarizerTest0() {
  TimeInScopesLogger summarizer("myTimeInScopesLogger");

  summarizer.start("first-stopwatch");
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  summarizer.stop();

  summarizer.start("un-autre-chronometre");
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  summarizer.stop();

  summarizer.start("first-stopwatch");
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  summarizer.stop();

  //  [myTimeInScopesLogger]
  //         first-stopwatch          : 0.005311 [s]     : 66.257112 %
  //         un-autre-chronometre     : 0.002646 [s]     : 33.015471 %
  //         unaccounteded time       : 0.000058 [s]     : 0.727417 %
  //         total time               : 0.008015 [s]     : 100.000000 %.

  summarizer.summarize();
}

// This is how the TimeInScopesLogger will be unsed in, for example, PopART. A
// global summarizer is constructed and accessed in the various parts of the
// codebase.
TimeInScopesLogger &summarizer() {
  static TimeInScopesLogger s("globby-globulus");
  return s;
}

void part0() {
  summarizer().start("part0");
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  summarizer().stop();
}

void part1() {
  summarizer().start("part1");
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  summarizer().stop();
}

void globalTest() {
  part0();
  part1();
  part0();
  part0();
  part1();

  const auto t0 = summarizer().get("part0");
  const auto t1 = summarizer().get("part1");

  // we test with 1 millisecond margin for error.
  if (t0 < 2e-3) {
    std::ostringstream oss;
    oss << "part0 ran for a total for 3 milliseconds, "
        << "incorrect time of " << t0 << '.';
    throw error(oss.str());
  }

  if (t1 < 3e-3) {
    std::ostringstream oss;
    oss << "part1 ran for a total for 4 milliseconds, "
        << "incorrect time of " << t1 << '.';
    throw error(oss.str());
  }

  summarizer().summarize();
}

void noDoubleStart() {

  TimeInScopesLogger s("foo");
  s.start("scope0");
  bool caught{false};
  try {
    s.start("scope1");
  } catch (const poprithms::error::error &err) {
    caught = true;
  }
  if (!caught) {
    throw error("Failed in test that start cannot be called without a stop");
  }
}

void doubleSameOk() {
  TimeInScopesLogger s("foo2");
  s.start("a");
  s.start("a");
  s.stop();
  s.start("a");
  s.stop();
  s.stop();
  s.start("a");
}

void timeRegisteredBeforeStop() {
  TimeInScopesLogger s("foo3");
  s.start("a");
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  if (s.get("a") < 1e-3) {
    throw error(
        "Stop should not be required to get accurate time measurement. ");
  }
}

// This is really a repeat test of Logger's functionality.
void noTwoTimeInScopesLoggersWithSameId() {

  TimeInScopesLogger a("new101");

  bool caught{false};
  try {
    TimeInScopesLogger b("new101");
  } catch (const poprithms::error::error &err) {
    caught = true;
  }

  if (!caught) {
    throw error("Failed to catch error when TimeInScopesLoggers of same "
                "names constructed");
  }
}

} // namespace

int main() {

  poprithms::logging::setGlobalLevel(poprithms::logging::Level::Info);
  summarizerTest0();
  globalTest();
  noDoubleStart();
  doubleSameOk();
  timeRegisteredBeforeStop();
  noTwoTimeInScopesLoggersWithSameId();
  poprithms::logging::setGlobalLevel(poprithms::logging::Level::Off);
}
