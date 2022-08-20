// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_LOGGING_TIMEPARTITIONLOGGER_HPP
#define POPRITHMS_LOGGING_TIMEPARTITIONLOGGER_HPP

#include <chrono>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <poprithms/logging/logging.hpp>
#include <poprithms/util/typedinteger.hpp>

namespace poprithms {
namespace logging {

using StopwatchId = poprithms::util::TypedInteger<'s', uint32_t>;

class TimePartitionLogger;

/**
 * A class following the RAII design pattern; when the class destructor is
 * called, the appropriate stopwatch of a TimePartitionLogger is stopped.
 * */
class ScopedStopwatch {

public:
  /** Custom desctructor, calls stop on stopwatch. */
  ~ScopedStopwatch();

  /** Moveable but not copyable. */
  ScopedStopwatch(ScopedStopwatch &&)            = default;
  ScopedStopwatch &operator=(ScopedStopwatch &&) = default;

  ScopedStopwatch(const ScopedStopwatch &)            = delete;
  ScopedStopwatch &operator=(const ScopedStopwatch &) = delete;

private:
  ScopedStopwatch(const std::string &, TimePartitionLogger &);
  std::unique_ptr<int> hasNotMoved;
  TimePartitionLogger *pLogger;
  friend class TimePartitionLogger;
};

/**
 * A TimePartitionLogger can be thought of as a set of stopwatches, where
 * there is never more than 1 stopwatch running at a time. Each stopwatch is
 * defined by a string.
 *
 * The TimePartitionLogger class might not behave as expected if invoked on
 * multiple threads in parallel.
 * */
class TimePartitionLogger {

public:
  TimePartitionLogger(const std::string &id)
      : timeOfConstruction(std::chrono::high_resolution_clock::now()),
        id_(id) {}
  virtual ~TimePartitionLogger() = default;

  std::string id() const { return id_; }

  /**
   * Start the stopwatch \a stopwatch. The behaviour when there is already a
   * stopwatch on depends on the inheriting class's implementation of \a
   * preHandleStartFromOn
   * */
  void start(const std::string &stopwatch);

  /**
   * Stop whichever stopwatch is currently on. The behaviour in the case where
   * the current stopwatch was activated when there was already a stopwatch on
   * depends on the inheriting class's implementation of \a
   * postHandleStartFromOn.
   * */
  void stop();

  /**
   * A summary of the times (the total cumulative time on the stopwatch in
   * seconds) and counts (the number of times the stopwatch was started) on
   * each stopwatch. An example might be:
   *
   *  Scope              Time [s]        Count  Percentage
   *  -----              --------        -----  ----------
   *  stopwatch-0        0.006390          100        40 %
   *  watch-1            0.004140           22        25 %
   *  another            0.003815            1        14 %
   *  Total              0.019593          n/a       100 %
   *  Accounted for      0.019486          n/a        99 %
   *  Unaccounted for    0.000107          n/a         1 %
   *
   *
   * */
  std::string str(double minPercentage) const;

  /**
   * Append summary string to output stream \a ost
   * \sa str
   * */
  void append(std::ostream &ost, double minPercentage) const;

  /**
   * Get the total time that stopwatch #stopwatch has been on for, in seconds.
   * This method is O(number-of-times stopwatches were switched on) and so
   * should be used sparingly.
   * */
  double get(const std::string &stopwatch) const;

  /**
   * Return the total elapsed time since this TimePartitionLogger was
   * constructed, in seconds.
   * */
  double sinceConstruction() const;

  /**
   * Return the cumulative time on all stopwatches, in seconds.
   * */
  double accounted() const;

  /**
   * Return the total amount of time which is not accounted for. That is,
   * the total amount of time since construction of this TimeInScopeLogger
   * when no stopwatches have been on.
   * */
  double unaccounted() const { return sinceConstruction() - accounted(); }

  /**
   * RAII programming technique to run a stopwatch for the duration that the
   * program being analyzed does not leave a certain scope. This can be
   * useful, for example, in functions which have several return points:
   *
   * <code>
   * int foo(){
   *  auto sw0 = myTimer().scopedStopwatch("sw0");
   *  if (condition0()) {
   *    return 0;
   *  }
   *  return 1;
   * }
   * </code>
   *
   * In this example, we don't need to worry about stopping the "sw0" at all
   * of the return sites, it will automatically stop timing when foo
   * returns.
   * */
  ScopedStopwatch scopedStopwatch(const std::string &stopwatch);

  /** The type of the events a stopwatch can perform */
  enum class EventType { Start, Stop } type;

  /**
   * An Event: when a stopwatch either starts of stops.
   * */
  struct Event {

    StopwatchId id;

    EventType type;

    // the (global) time of the event
    using TP = std::chrono::high_resolution_clock::time_point;
    TP time_;

    // the name of the stopwatch
    // const std::string & stopwatch() const;

    bool isStart() const { return type == EventType::Start; }
    bool isStop() const { return type == EventType::Stop; }

    Event(StopwatchId id_, EventType t_) : id(id_), type(t_) {}
    Event(StopwatchId id_, EventType t_, TP tm_)
        : id(id_), type(t_), time_(tm_) {}
  };

  using Events = std::vector<Event>;

public:
  /** \return true if there is currently a stopwatch which is on. */
  bool isOn() const { return !events_.empty() && events_.back().isStart(); }

  /** \return true if currently all stopwatches are off. */
  bool isOff() const { return !isOn(); }

  /** \return A vector of all the Events registered. */
  const std::vector<Event> &events() const { return events_; }

  /** \return A string of all the Events registered. */
  std::string eventsStr() const;

  /**
   * For testing purposes: verify that all the Events registered match the
   * Events in \a expected, excluding the times of Events.
   */
  using TimelessEvent  = std::pair<std::string, EventType>;
  using TimelessEvents = std::vector<TimelessEvent>;
  void verifyEvents(const TimelessEvents &expected) const;

  /**
   * How long has the current stopwatch been on for, if any stopwatches are
   * on, in seconds.
   * */
  double beenOnFor() const;

  /**
   * Return the stopwatch which is currently running, if there is one.
   * Otherwise throw an error.
   * */
  std::string currentStopwatch() const;

protected:
  void registerStartEvent(const std::string &stopwatch);
  void registerStopEvent();

private:
  using TimePoint = std::chrono::high_resolution_clock::time_point;

  std::vector<Event> completeAndGet() const;

  TimePoint timeOfConstruction;

  /**
   * Handle the case of starting a stopwatch, when there is already one on.
   */
  virtual void preHandleStartFromOn(const std::string &stopwatch) = 0;

  /**
   * Handle the case of stopping a stopwatch, when the stopwatch being
   * stopped was started when another stopwatch was on.
   */
  virtual void postHandleStartFromOn() = 0;

  std::vector<Event> events_;

  std::string id_;

  // Mapping between the names of stopwatches, and a StopwatchId.
  // Each stopwatch (string) will have a unique StopwatchId, and the
  // StopwatchIds will be contiguous from 0.
  std::unordered_map<std::string, StopwatchId> stopwatchIds;
  std::vector<std::string> stopwatchNames;

  // Retrieve the StopwatchId for the string stopwatch. If 'stopwatch' has not
  // been seen before, and error will be thrown. See also createStopwatchId.
  StopwatchId stopwatchId(const std::string &stopwatch) const;

  // Retrieve the StopwatchId for the string stopwatch. If 'stopwatch' has not
  // been seen before, a new StopwatchId is created and returned. See also
  // createStopwatch.
  StopwatchId createStopwatchId(const std::string &stopwatch);

  // Retrieve stopwatch (name) of 'id'. If 'id' is not valid, an error is
  // thrown.
  std::string stopwatch(StopwatchId id) const;

  virtual void noWeakVTables();
};

/**
 * Throw an error if a new stopwatch is started before stopping the current
 * one.
 * */
class ManualTimePartitionLogger : public TimePartitionLogger {
public:
  explicit ManualTimePartitionLogger(const std::string &id)
      : TimePartitionLogger(id) {}

  ManualTimePartitionLogger() : ManualTimePartitionLogger(std::string{}) {}

private:
  void preHandleStartFromOn(const std::string &stopwatch) final;
  void postHandleStartFromOn() final {}
};

/**
 * Put the current stopwatch on hold when a new stopwatch is started, and
 * restart it when the new stopwatch stops.
 * */

class SwitchingTimePartitionLogger : public TimePartitionLogger {
public:
  explicit SwitchingTimePartitionLogger(const std::string &id)
      : TimePartitionLogger(id) {}

  SwitchingTimePartitionLogger()
      : SwitchingTimePartitionLogger(std::string{}) {}

private:
  void preHandleStartFromOn(const std::string &stopwatch) final;
  void postHandleStartFromOn() final;

  std::vector<std::string> onHoldStack;
};

std::ostream &operator<<(std::ostream &,
                         const TimePartitionLogger::EventType &);

} // namespace logging
} // namespace poprithms

#endif
