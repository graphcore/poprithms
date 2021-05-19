// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <ostream>
#include <sstream>

#include <poprithms/logging/error.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace logging {

using Event  = TimePartitionLogger::Event;
using Events = std::vector<TimePartitionLogger::Event>;

ScopedStopwatch::ScopedStopwatch(const std::string &s,
                                 TimePartitionLogger &logger)
    : hasNotMoved(std::make_unique<int>()), pLogger(&logger) {
  logger.start(s);
}

ScopedStopwatch::~ScopedStopwatch() {
  if (hasNotMoved) {
    pLogger->stop();
  }
}

ScopedStopwatch
TimePartitionLogger::scopedStopwatch(const std::string &stopwatch) {
  return ScopedStopwatch(stopwatch, *this);
}

double TimePartitionLogger::beenOnFor() const {
  if (isOff()) {
    return 0.;
  }
  const auto stop_          = std::chrono::high_resolution_clock::now();
  using TimeInterval        = std::chrono::duration<double>;
  const TimeInterval delta_ = (stop_ - lastEventTime);
  const double delta        = delta_.count();
  return delta;
}

std::string TimePartitionLogger::eventsStr() const {

  uint64_t maxScopeLength{0ull};
  for (const auto &[scope, delta] : stopwatches) {
    (void)delta;
    maxScopeLength = std::max<uint64_t>(maxScopeLength, scope.size());
  }

  std::ostringstream oss;
  oss << "Events of " << id() << ':';
  for (const auto &event : events()) {
    oss << "\n       "
        << (event.type == Event::Type::Start ? "Start " : "Stop  ")
        << event.stopwatch
        << util::spaceString(maxScopeLength + 2, event.stopwatch) << " : "
        << event.sinceConstruction;
  }
  oss << '.';
  return oss.str();
}

namespace {

using TimelessEvent  = std::pair<std::string, Event::Type>;
using TimelessEvents = std::vector<TimelessEvent>;
void appendTimelessEvent(std::ostream &ost, const TimelessEvents &v) {
  for (auto x : v) {
    ost << "\n        "
        << (x.second == Event::Type::Start ? "Start " : "Stop  ") << x.first;
  }
}

TimelessEvents getTimelessEvent(const Events &events) {
  TimelessEvents ts;
  ts.reserve(events.size());
  for (const auto &e : events) {
    ts.push_back({e.stopwatch, e.type});
  }
  return ts;
}

} // namespace

void TimePartitionLogger::verifyEvents(const TimelessEvents &expected) const {

  auto timelesses = getTimelessEvent(events_);
  auto getString  = [&timelesses, &expected]() {
    std::ostringstream oss;
    oss << "\nFailed in verifyEvents. Expected";
    appendTimelessEvent(oss, expected);
    oss << "\nObserved";
    appendTimelessEvent(oss, timelesses);
    return oss.str();
  };
  if (expected != timelesses) {
    throw error(getString());
  }
}

void TimePartitionLogger::increment(std::map<std::string, double> &m,
                                    const std::string &k,
                                    double t) const {
  auto found = m.find(k);
  if (found == m.end()) {
    m.insert({k, t});
  } else {
    found->second += t;
  }
}

double TimePartitionLogger::sinceConstruction() const {

  // Total time since creation of this TimePartitionLogger:
  const std::chrono::duration<double> total_ =
      std::chrono::high_resolution_clock::now() - constructionTime;
  const auto total = total_.count();
  return total;
}

double TimePartitionLogger::accounted() const {

  // Check how much time is on all the stopwatches combined:
  double totalAccounted{0.};
  for (const auto &[scope, delta] : stopwatches) {
    (void)scope;
    totalAccounted += delta;
  }

  // And add any time not logged:
  if (isOn()) {
    totalAccounted += beenOnFor();
  }
  return totalAccounted;
}

void TimePartitionLogger::summarizeInfo(double minPercentage) const {
  summarize(minPercentage, Level::Info);
}

void TimePartitionLogger::append(std::ostream &ost,
                                 double minPercentage) const {

  std::map<std::string, double> stopwatchesCopy = stopwatches;

  // Make a final increment to the stopwatch which is currently on (if there
  // is one which is currently on).
  if (isOn()) {
    increment(stopwatchesCopy, currentStopwatch(), beenOnFor());
  }

  std::map<std::string, std::string> stopwatchTimeStrings;
  std::map<std::string, double> stopwatchPercs;

  const auto total = sinceConstruction();

  auto ensureUnique = [this](std::string x) {
    while (stopwatches.count(x) > 0) {
      x = x + ' ';
    }
    return x;
  };
  const std::string totalTime       = ensureUnique("Total");
  const std::string unaccountedTime = ensureUnique("Unaccounted for");

  stopwatchesCopy.emplace(totalTime, total);
  stopwatchesCopy.emplace(unaccountedTime, unaccounted());

  // this vector will collect all the entries to be logged, then sort from
  // longest time to shortest time:
  std::vector<std::pair<double, std::string>> timesAndScopes;

  // For aligned logging, we get the longest stopwatch name and time
  // representation:
  uint64_t maxScopeLength{0};
  uint64_t maxTimeLength{0};
  for (const auto &[scope, delta] : stopwatchesCopy) {
    const auto perc    = 100. * delta / total;
    const auto timeStr = std::to_string(delta) + " [s]";
    if (perc >= minPercentage || scope == totalTime ||
        scope == unaccountedTime) {
      stopwatchTimeStrings.emplace(scope, timeStr);
      stopwatchPercs.emplace(scope, perc);
      maxScopeLength = std::max<uint64_t>(maxScopeLength, scope.size());
      maxTimeLength  = std::max<uint64_t>(maxTimeLength, timeStr.size());
      timesAndScopes.push_back({delta, scope});
    }
  }

  auto append = [maxScopeLength,
                 maxTimeLength,
                 &ost,
                 &stopwatchPercs,
                 &stopwatchTimeStrings](const std::string &scope) {
    const auto timeStr = stopwatchTimeStrings.at(scope);
    const auto perc    = stopwatchPercs.at(scope);
    const auto percStr = std::to_string(int(perc));

    ost << "       " << scope << util::spaceString(maxScopeLength + 2, scope)
        << " : " << timeStr << util::spaceString(maxTimeLength + 2, timeStr)
        << " :" << util::spaceString(5, percStr) << percStr << " %";
  };

  std::sort(timesAndScopes.rbegin(), timesAndScopes.rend());

  for (auto timeScope : timesAndScopes) {
    const auto scope = timeScope.second;
    if (scope != totalTime && scope != unaccountedTime) {
      if (stopwatchPercs.at(scope) >= minPercentage) {
        append(scope);
        ost << '\n';
      }
    }
  }

  append(unaccountedTime);
  ost << '\n';
  append(totalTime);
  ost << '.';
}

double TimePartitionLogger::get(const std::string &s) const {
  const auto found = stopwatches.find(s);
  if (found == stopwatches.cend()) {
    throw error("No stopwatch named " + s +
                ", error in TimePartitionLogger::get. ");
  }

  auto t0 = found->second;
  if (isOn() && currentStopwatch() == s) {
    t0 += beenOnFor();
  }

  return t0;
}

std::string TimePartitionLogger::str(double minPercentage) const {
  std::ostringstream ost;
  append(ost, minPercentage);
  return ost.str();
}

void TimePartitionLogger::summarize(double minPercentage, Level l) const {

  auto current         = getLevel();
  const auto shouldLog = static_cast<int>(current) <= static_cast<int>(l);
  if (shouldLog) {
    std::cout << str(minPercentage) << std::endl;
  }
}

void TimePartitionLogger::registerStartEvent(const std::string &stopwatch) {
  std::lock_guard<std::mutex> m(mut);

  clockState        = ClockState::On;
  currentStopwatch_ = stopwatch;
  lastEventTime     = std::chrono::high_resolution_clock::now();
  increment(0.);
  events_.push_back({stopwatch, Event::Type::Start, sinceConstruction()});
}

void TimePartitionLogger::registerStopEvent() {
  std::lock_guard<std::mutex> m(mut);
  if (isOff()) {
    throw error("Cannot register stop event when all stopwatches are off");
  }
  increment(beenOnFor());
  events_.push_back(
      {currentStopwatch(), Event::Type::Stop, sinceConstruction()});
  clockState        = ClockState::Off;
  currentStopwatch_ = "none";
  lastEventTime     = std::chrono::high_resolution_clock::now();
}

std::string TimePartitionLogger::currentStopwatch() const {
  if (isOff()) {
    std::ostringstream oss;
    oss << "Invalid call TimePartitionLogger::currentStopwatch, as "
        << "there are currently no stopwatches on. ";
    throw error(oss.str());
  }
  return currentStopwatch_;
}

void TimePartitionLogger::start(const std::string &stopwatch) {
  if (isOn()) {
    preHandleStartFromOn(stopwatch);
  }
  registerStartEvent(stopwatch);
}

void ManualTimePartitionLogger::preHandleStartFromOn(
    const std::string &stopwatch) {
  // It is not ok to start a stopwatch if there is another one already
  // running. We decided to use the TimePartitionLogger class to partition the
  // total time, and are going stick to this decision (i.e. no hierarchy of
  // calls recorded).
  std::ostringstream oss;
  oss << "Invalid call, ManualTimePartitionLogger::start(" << stopwatch
      << "),  as this TimePartitionLogger "
      << "is currently running stopwatch `" << currentStopwatch()
      << "`. stop() should be called before changing "
      << "to stopwatch, `" << stopwatch << "`. ";
  throw error(oss.str());
}

void TimePartitionLogger::stop() {
  registerStopEvent();
  postHandleStartFromOn();
}

void SwitchingTimePartitionLogger::preHandleStartFromOn(const std::string &) {
  onHoldStack.push_back(currentStopwatch());
  registerStopEvent();
}

void SwitchingTimePartitionLogger::postHandleStartFromOn() {
  if (!onHoldStack.empty()) {
    registerStartEvent(onHoldStack.back());
    onHoldStack.pop_back();
  }
}

} // namespace logging
} // namespace poprithms
