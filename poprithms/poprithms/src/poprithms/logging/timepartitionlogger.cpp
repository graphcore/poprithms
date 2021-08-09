// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <logging/error.hpp>
#include <map>
#include <mutex>
#include <numeric>
#include <ostream>
#include <sstream>
#include <unordered_set>

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
  if (!events_.empty() && events_.back().isStart()) {
    const auto stop_ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> delta_ = stop_ - events_.back().time_;
    return delta_.count();
  }
  return 0.;
}

std::string TimePartitionLogger::eventsStr() const {

  uint64_t maxStopwatchLength{0ull};
  for (const auto &e : events_) {
    maxStopwatchLength =
        std::max<uint64_t>(maxStopwatchLength, e.stopwatch.size());
  }

  std::ostringstream oss;
  oss << "Events of " << id() << ':';
  for (const auto &event : events()) {

    const std::chrono::duration<double> dt = event.time_ - timeOfConstruction;
    oss << "\n       " << event.type << event.stopwatch
        << util::spaceString(maxStopwatchLength + 2, event.stopwatch) << " : "
        << dt.count();
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

TimelessEvents getTimelessEvents(const Events &events) {
  TimelessEvents ts;
  ts.reserve(events.size());
  for (const auto &e : events) {
    ts.push_back({e.stopwatch, e.type});
  }
  return ts;
}

} // namespace

void TimePartitionLogger::verifyEvents(const TimelessEvents &expected) const {

  auto timelesses = getTimelessEvents(events_);
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

double TimePartitionLogger::sinceConstruction() const {

  // Total time since creation of this TimePartitionLogger:
  const std::chrono::duration<double> total_ =
      std::chrono::high_resolution_clock::now() - timeOfConstruction;
  const auto total = total_.count();
  return total;
}

namespace {

/**
 * Class for summarizing all the uses of stopwatch, including
 * (1) the total time the stopwatch is on, and
 * (2) (optionally) the total number of times the stopwatch is turned on.
 * */
class StopwatchSummary {
public:
  // Construct a time-only summary entry
  explicit StopwatchSummary(const std::string &sw, double t)
      : stopwatch_(sw), time_(t), hasCount_(false) {}

  // Constuct a time+count summary entry
  explicit StopwatchSummary(const std::string &sw, double t, uint64_t n)
      : stopwatch_(sw), time_(t), count_(n), hasCount_(true) {}

  double time() const { return time_; }

  bool hasCount() const { return hasCount_; }

  std::string timeStr() const { return std::to_string(time_) + "       "; }

  std::string countStr() const {
    return hasCount() ? std::to_string(count_) : "n/a";
  }

  std::string percStr(double d) const {
    return std::to_string(static_cast<int>(0.5 + 100. * time_ / d)) + " %";
  }

  void incrementTime(double d) { time_ += d; }

  void incrementCount(uint64_t d) { count_ += d; }

  bool operator<(const StopwatchSummary &rhs) const {
    return tup() < rhs.tup();
  }

  bool operator==(const StopwatchSummary &rhs) const {
    return tup() == rhs.tup();
  }

  const std::string &stopwatch() const { return stopwatch_; }

private:
  std::tuple<bool, double, std::string, uint64_t> tup() const {
    return {hasCount_, time_, stopwatch_, count_};
  }
  std::string stopwatch_;
  double time_;
  uint64_t count_;
  bool hasCount_;
};

class StopwatchSummaries {

public:
  /**
   * Construct a summary of all the stopwatches encountered in a sequence of
   * Events.
   * */
  StopwatchSummaries(const std::vector<Event> &events) {

    if (events.size() % 2 != 0) {
      throw error(
          "Expected an even number of events: one stop for every start.");
    }
    for (uint64_t i = 0; i < events.size() / 2; ++i) {
      auto j = 2 * i;
      if (!events[j].isStart() || events[j + 1].isStart()) {
        throw error("Expected events to be alternating starts and stops. ");
      }
      if (events[j].stopwatch != events[j + 1].stopwatch) {
        throw error(
            "Expected start-stop pairs to be on a single stopwatch. ");
      }

      auto stopwatch = events[j].stopwatch;

      // time between start and stop.
      const std::chrono::duration<double> dt =
          events[j + 1].time_ - events[j].time_;
      auto t0    = dt.count();
      auto found = summaries.find(stopwatch);
      if (found == summaries.cend()) {
        summaries.insert({stopwatch, StopwatchSummary(stopwatch, t0, 1ULL)});
      } else {
        found->second.incrementCount(1ULL);
        found->second.incrementTime(t0);
      }
    }
  }

  /**
   * Insert a (time-only) stopwatch.
   * */
  void insert(const std::string &stopwatch, double t) {
    summaries.insert({stopwatch, StopwatchSummary(stopwatch, t)});
  }

  /**
   * The sum of all the times on all of the stopwatches
   * */
  double totalAccountedFor() const {
    return std::accumulate(
        summaries.cbegin(),
        summaries.cend(),
        0.0,
        [](double v, const auto &x) { return v + x.second.time(); });
  }

  const auto &get() const { return summaries; }

private:
  std::map<std::string, StopwatchSummary> summaries;
};

} // namespace

double TimePartitionLogger::get(const std::string &stopwatch) const {
  const auto m     = StopwatchSummaries(completeAndGet()).get();
  const auto found = m.find(stopwatch);
  if (found == m.cend()) {
    return 0;
  }
  return found->second.time();
}

std::vector<Event> TimePartitionLogger::completeAndGet() const {

  auto eventsCopy_ = events_;

  // Make a final increment to the time on the stopwatch which is currently on
  // (if there is one which is currently on).
  const auto finalTime = std::chrono::high_resolution_clock::now();
  if (!eventsCopy_.empty() && eventsCopy_.back().isStart()) {
    eventsCopy_.push_back(
        {eventsCopy_.back().stopwatch, Event::Type::Stop, finalTime});
  }

  return eventsCopy_;
}

void TimePartitionLogger::append(std::ostream &ost,
                                 double minPercentage) const {

  auto eventsCopy_ = completeAndGet();

  std::unordered_set<std::string> allStopwatches;
  for (const auto &e : eventsCopy_) {
    allStopwatches.insert(e.stopwatch);
  }

  auto ensureUnique = [allStopwatches](std::string x) {
    while (allStopwatches.count(x) > 0) {
      x = x + '_';
    }
    return x;
  };
  const std::string totalTime       = ensureUnique("Total");
  const std::string unaccountedTime = ensureUnique("Unaccounted for   ");
  const std::string accountedTime   = ensureUnique("Accounted for");

  auto summaries    = StopwatchSummaries(eventsCopy_);
  auto accountedFor = summaries.totalAccountedFor();
  const auto total  = sinceConstruction();
  summaries.insert(totalTime, total);
  summaries.insert(accountedTime, accountedFor);
  summaries.insert(unaccountedTime, total - accountedFor);

  // elements of the columns of the summary string:
  std::vector<std::string> scopes;
  std::vector<std::string> times;
  std::vector<std::string> counts;
  std::vector<std::string> percentages;

  std::vector<StopwatchSummary> v;
  for (const auto &[stopwatch, summary] : summaries.get()) {
    (void)stopwatch;
    v.push_back(summary);
  }
  std::sort(v.rbegin(), v.rend());

  for (const auto &summary : v) {
    auto perc = (100.0 * summary.time()) / total;
    if (perc >= minPercentage || !summary.hasCount()) {
      scopes.push_back(summary.stopwatch());
      times.push_back(summary.timeStr());
      counts.push_back(summary.countStr());
      percentages.push_back(summary.percStr(total));
    }
  }

  ost << util::alignedColumns(
      {{"Scope", scopes},
       {"Time [s]", times},
       {"Count", counts, '-', util::StringColumn::Align::Right},
       {"Percentage", percentages, '-', util::StringColumn::Align::Right}});
}

std::string TimePartitionLogger::str(double minPercentage) const {
  std::ostringstream ost;
  append(ost, minPercentage);
  return ost.str();
}

void TimePartitionLogger::registerStartEvent(const std::string &stopwatch) {
  events_.push_back({stopwatch,
                     Event::Type::Start,
                     std::chrono::high_resolution_clock::now()});
}

void TimePartitionLogger::registerStopEvent() {
  if (isOff()) {
    throw error("Cannot register stop event when all stopwatches are off");
  }
  events_.push_back({events_.back().stopwatch,
                     Event::Type::Stop,
                     std::chrono::high_resolution_clock::now()});
}

std::string TimePartitionLogger::currentStopwatch() const {
  if (events_.empty() || !events_.back().isStart()) {
    std::ostringstream oss;
    oss << "Invalid call TimePartitionLogger::currentStopwatch, as "
        << "there are currently no stopwatches on. ";
    throw error(oss.str());
  }
  return events_.back().stopwatch;
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

std::ostream &operator<<(std::ostream &ost,
                         const TimePartitionLogger::Event::Type &t) {
  switch (t) {
  case (Event::Type::Start): {
    ost << "Start";
    return ost;
  }
  case (Event::Type::Stop): {
    ost << "Stop";
    return ost;
  }
  }
  throw error("Unhandled Event::Type");
}

} // namespace logging
} // namespace poprithms
