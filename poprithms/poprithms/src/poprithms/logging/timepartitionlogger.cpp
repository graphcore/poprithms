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
using Events = TimePartitionLogger::Events;

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
        std::max<uint64_t>(maxStopwatchLength, stopwatch(e.id).size());
  }

  std::ostringstream oss;
  oss << "Events of " << id() << ':';
  for (const auto &event : events()) {

    const std::chrono::duration<double> dt = event.time_ - timeOfConstruction;
    oss << "\n       " << event.type << stopwatch(event.id)
        << util::spaceString(maxStopwatchLength + 2, stopwatch(event.id))
        << " : " << dt.count();
  }
  oss << '.';
  return oss.str();
}

namespace {

void appendTimelessEvent(std::ostream &ost,
                         const TimePartitionLogger::TimelessEvents &v) {
  for (auto x : v) {
    ost << "\n        "
        << (x.second == TimePartitionLogger::EventType::Start ? "Start "
                                                              : "Stop  ")
        << x.first;
  }
}

} // namespace

void TimePartitionLogger::verifyEvents(const TimelessEvents &expected) const {

  auto getTimelessEvents = [this](const Events &events) {
    TimePartitionLogger::TimelessEvents ts;
    ts.reserve(events.size());
    for (const auto &e : events) {
      ts.push_back({stopwatch(e.id), e.type});
    }
    return ts;
  };

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
  // Constuct a time+count summary entry
  explicit StopwatchSummary(double t, uint64_t n) : time_(t), count_(n) {}

  double time() const { return time_; }

  std::string timeStr() const { return std::to_string(time_) + "       "; }

  std::string countStr() const { return std::to_string(count_); }

  static std::string percStr(double n, double d) {
    return std::to_string(static_cast<int>(0.5 + 100. * n / d)) + " %";
  }

  std::string percStr(double d) const { return percStr(time_, d); }

  void incrementTime(double d) { time_ += d; }

  void incrementCount(uint64_t d) { count_ += d; }

  bool operator<(const StopwatchSummary &rhs) const {
    return tup() < rhs.tup();
  }

  bool operator==(const StopwatchSummary &rhs) const {
    return tup() == rhs.tup();
  }

private:
  std::tuple<double, uint64_t> tup() const { return {time_, count_}; }
  double time_;
  uint64_t count_;
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
      if (events[j].id != events[j + 1].id) {
        throw error(
            "Expected start-stop pairs to be on a single stopwatch. ");
      }

      auto stopwatch = events[j].id;

      // time between start and stop.
      const std::chrono::duration<double> dt =
          events[j + 1].time_ - events[j].time_;
      auto t0 = dt.count();
      if (summaries.size() <= stopwatch.get()) {
        summaries.resize(stopwatch.get() + 1, StopwatchSummary(0., 0));
      }
      summaries[stopwatch.get()].incrementCount(1ULL);
      summaries[stopwatch.get()].incrementTime(t0);
    }
  }

  /**
   * The sum of all the times on all of the stopwatches
   * */
  double totalAccountedFor() const {
    return std::accumulate(
        summaries.cbegin(),
        summaries.cend(),
        0.0,
        [](double v, const auto &x) { return v + x.time(); });
  }

  uint64_t nStopwatches() const { return summaries.size(); }

  const StopwatchSummary &getSummary(StopwatchId id) const {
    if (id.get() >= summaries.size()) {
      throw error("Invalid StopwatchId");
    }
    return summaries[id.get()];
  }

private:
  std::vector<StopwatchSummary> summaries;
};

} // namespace

double TimePartitionLogger::get(const std::string &stopwatch) const {
  StopwatchId id = stopwatchId(stopwatch);
  return StopwatchSummaries(completeAndGet()).getSummary(id).time();
}

std::vector<Event> TimePartitionLogger::completeAndGet() const {

  auto eventsCopy_ = events_;

  // Make a final increment to the time on the stopwatch which is currently on
  // (if there is one which is currently on).
  const auto finalTime = std::chrono::high_resolution_clock::now();
  if (!eventsCopy_.empty() && eventsCopy_.back().isStart()) {
    eventsCopy_.push_back(
        {eventsCopy_.back().id, EventType::Stop, finalTime});
  }

  return eventsCopy_;
}

void TimePartitionLogger::append(std::ostream &ost,
                                 double minPercentage) const {

  auto eventsCopy_ = completeAndGet();

  std::unordered_set<std::string> allStopwatches;
  for (const auto &e : eventsCopy_) {
    allStopwatches.insert(stopwatch(e.id));
  }

  auto ensureUnique = [allStopwatches](std::string x) {
    while (allStopwatches.count(x) > 0) {
      x = x + '_';
    }
    return x;
  };

  auto summaries    = StopwatchSummaries(eventsCopy_);
  auto accountedFor = summaries.totalAccountedFor();
  const auto total  = sinceConstruction();

  std::vector<std::pair<std::string, double>> additionals{
      {ensureUnique("Total"), total},
      {ensureUnique("Unaccounted for   "), total - accountedFor},
      {ensureUnique("Accounted for"), accountedFor}};

  // elements of the columns of the summary string:
  std::vector<std::string> scopes;
  std::vector<std::string> times;
  std::vector<std::string> counts;
  std::vector<std::string> percentages;

  std::vector<std::pair<double, StopwatchId>> ts;

  for (uint64_t i = 0; i < summaries.nStopwatches(); ++i) {
    ts.push_back({summaries.getSummary(i).time(), i});
  }
  std::sort(ts.rbegin(), ts.rend());

  for (auto t : ts) {
    auto id             = t.second;
    const auto &summary = summaries.getSummary(id);
    auto perc           = (100.0 * summary.time()) / total;
    if (perc >= minPercentage) {
      scopes.push_back(stopwatch(id));
      times.push_back(summary.timeStr());
      counts.push_back(summary.countStr());
      percentages.push_back(summary.percStr(total));
    }
  }

  for (auto a : additionals) {
    scopes.push_back(a.first);
    times.push_back(std::to_string(a.second));
    counts.push_back("n/a");
    percentages.push_back(StopwatchSummary::percStr(a.second, total));
  }

  using Parameters = poprithms::util::StringColumn::Parameters;
  ost << util::alignedColumns(
      {{"Scope", scopes, Parameters()},
       {"Time [s]", times, Parameters()},
       {"Count",
        counts,
        Parameters().alignType(util::StringColumn::Align::Right)},
       {"Percentage",
        percentages,
        Parameters().alignType(util::StringColumn::Align::Right)}});
}

std::string TimePartitionLogger::str(double minPercentage) const {
  std::ostringstream ost;
  append(ost, minPercentage);
  return ost.str();
}

void TimePartitionLogger::registerStartEvent(const std::string &stopwatch) {

  events_.push_back({createStopwatchId(stopwatch),
                     EventType::Start,
                     std::chrono::high_resolution_clock::now()});
}

StopwatchId TimePartitionLogger::createStopwatchId(const std::string &name) {
  auto found = stopwatchIds.find(name);
  if (found != stopwatchIds.cend()) {
    return found->second;
  }
  StopwatchId nxt = stopwatchNames.size();
  stopwatchIds.insert({name, nxt.get()});
  stopwatchNames.push_back(name);
  return nxt;
}

StopwatchId TimePartitionLogger::stopwatchId(const std::string &name) const {
  auto found = stopwatchIds.find(name);
  if (found != stopwatchIds.cend()) {
    return found->second;
  }
  throw error("Invalid stopwatch name " + name);
}

void TimePartitionLogger::registerStopEvent() {

  if (isOff()) {
    throw error("Cannot register stop event when all stopwatches are off");
  }
  events_.push_back({events_.back().id,
                     EventType::Stop,
                     std::chrono::high_resolution_clock::now()});
}

std::string TimePartitionLogger::currentStopwatch() const {
  if (events_.empty() || !events_.back().isStart()) {
    std::ostringstream oss;
    oss << "Invalid call TimePartitionLogger::currentStopwatch, as "
        << "there are currently no stopwatches on. ";
    throw error(oss.str());
  }
  return stopwatch(events_.back().id);
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
                         const TimePartitionLogger::EventType &t) {
  switch (t) {
  case (TimePartitionLogger::EventType::Start): {
    ost << "Start";
    return ost;
  }
  case (TimePartitionLogger::EventType::Stop): {
    ost << "Stop";
    return ost;
  }
  }
  throw error("Unhandled EventType");
}

std::string TimePartitionLogger::stopwatch(StopwatchId id) const {
  return stopwatchNames.at(id.get());
}

} // namespace logging
} // namespace poprithms
