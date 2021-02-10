// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <sstream>

#include <poprithms/logging/error.hpp>
#include <poprithms/logging/timeinscopeslogger.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace logging {

void TimeInScopesLogger::start(const std::string &stopwatch) {

  // It's ok to start the stopwatch which is currently running, this is a
  // no-op.
  if (currentStopwatch == stopwatch) {
    return;
  }

  // It is not ok to start a stopwatch if there is another one already
  // running. We decided use the TimeInScopesLogger class to partition the
  // total time, gonna stick to this decision.
  if (on()) {
    std::ostringstream oss;
    oss << "Invalid call, TimeInScopesLogger::start(" << stopwatch
        << "),  as "
        << "this TimeInScopesLogger is currently running the stopwatch "
        << currentStopwatch
        << ". TimeInScopesLogger::stop() should be called before changing "
        << "to stopwatch, `" << stopwatch << "`. ";
    throw error(oss.str());
  }

  // Ok, request to start a new stopwatch accepted:
  clockState              = ClockState::On;
  currentStopwatch        = stopwatch;
  startOfCurrentStopwatch = std::chrono::high_resolution_clock::now();
  increment(stopwatches, currentStopwatch, 0.);
}

double TimeInScopesLogger::sinceCurrentStart() const {
  if (off()) {
    return 0.;
  }
  const auto stop_          = std::chrono::high_resolution_clock::now();
  using TimeInterval        = std::chrono::duration<double>;
  const TimeInterval delta_ = (stop_ - startOfCurrentStopwatch);
  const double delta        = delta_.count();
  return delta;
}

void TimeInScopesLogger::stop() {

  // calling stop multiple times in a row is fine
  if (off()) {
    return;
  }

  increment(stopwatches, currentStopwatch, sinceCurrentStart());
  clockState       = ClockState::Off;
  currentStopwatch = "none";
}

void TimeInScopesLogger::increment(std::map<std::string, double> &m,
                                   const std::string &k,
                                   double t) const {
  auto found = m.find(k);
  if (found == m.end()) {
    m.insert({k, t});
  } else {
    found->second += t;
  }
}

double TimeInScopesLogger::sinceConstruction() const {

  // Total time since creation of this TimeInScopesLogger:
  const std::chrono::duration<double> total_ =
      std::chrono::high_resolution_clock::now() - constructionTime;
  const auto total = total_.count();
  return total;
}

double TimeInScopesLogger::accounted() const {

  // Check how much time is on all the stopwatches combined:
  double totalAccounted{0.};
  for (const auto &[scope, delta] : stopwatches) {
    (void)scope;
    totalAccounted += delta;
  }

  // And add any time not logged:
  if (on()) {
    totalAccounted += sinceCurrentStart();
  }
  return totalAccounted;
}

void TimeInScopesLogger::append(std::ostream &ost) const {

  std::map<std::string, double> stopwatchesCopy = stopwatches;
  if (on()) {
    increment(stopwatchesCopy, currentStopwatch, sinceCurrentStart());
  }

  std::map<std::string, std::string> stopwatchTimeStrings;
  std::map<std::string, std::string> stopwatchPercStrings;

  const auto total = sinceConstruction();

  const std::string totalTime{"total"};
  const std::string unaccountedTime{"unaccounted"};

  stopwatchesCopy.emplace(totalTime, total);
  stopwatchesCopy.emplace(unaccountedTime, unaccounted());

  // For aligned logging, we get the longest stopwatch name and time
  // representation:
  uint64_t maxScopeLength{0};
  uint64_t maxTimeLength{0};
  for (const auto &[scope, delta] : stopwatchesCopy) {
    const auto timeStr = std::to_string(delta) + " [s]";
    stopwatchTimeStrings.emplace(scope, timeStr);
    stopwatchPercStrings.emplace(
        scope, std::to_string(static_cast<int>(100. * delta / total)));
    maxScopeLength = std::max<uint64_t>(maxScopeLength, scope.size());
    maxTimeLength  = std::max<uint64_t>(maxTimeLength, timeStr.size());
  }

  ost << "[" << id() << "]\n";
  auto append = [maxScopeLength,
                 maxTimeLength,
                 &ost,
                 &stopwatchPercStrings,
                 &stopwatchTimeStrings](const std::string &scope) {
    const auto timeStr = stopwatchTimeStrings.at(scope);
    const auto percStr = stopwatchPercStrings.at(scope);

    ost << "       " << scope << util::spaceString(maxScopeLength + 2, scope)
        << " : " << timeStr << util::spaceString(maxTimeLength + 2, timeStr)
        << " :" << util::spaceString(5, percStr) << percStr << " %";
  };

  for (const auto &[scope, timeStr] : stopwatchTimeStrings) {
    (void)timeStr;
    if (scope != totalTime && scope != unaccountedTime) {
      append(scope);
      ost << '\n';
    }
  }
  append(unaccountedTime);
  ost << '\n';
  append(totalTime);
  ost << '.';
}

double TimeInScopesLogger::get(const std::string &s) const {
  const auto found = stopwatches.find(s);
  if (found == stopwatches.cend()) {
    throw error("No stopwatch named " + s + ", error in getSeconds. ");
  }

  auto t0 = found->second;
  if (on() && currentStopwatch == s) {
    t0 += sinceCurrentStart();
  }

  return t0;
}

std::string TimeInScopesLogger::str() const {
  std::ostringstream ost;
  append(ost);
  return ost.str();
}

void TimeInScopesLogger::summarize(Level l) const {

  auto current         = getLevel();
  const auto shouldLog = static_cast<int>(current) <= static_cast<int>(l);
  if (shouldLog) {
    std::cout << str() << std::endl;
  }
}

} // namespace logging
} // namespace poprithms
