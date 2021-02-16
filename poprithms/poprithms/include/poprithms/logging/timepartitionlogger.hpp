// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_LOGGING_TIMEPARTITIONLOGGER_HPP
#define POPRITHMS_LOGGING_TIMEPARTITIONLOGGER_HPP

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include <poprithms/logging/logging.hpp>

namespace poprithms {
namespace logging {

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
  ScopedStopwatch(ScopedStopwatch &&) = default;
  ScopedStopwatch &operator=(ScopedStopwatch &&) = default;

  ScopedStopwatch(const ScopedStopwatch &) = delete;
  ScopedStopwatch &operator=(const ScopedStopwatch &) = delete;

private:
  ScopedStopwatch(const std::string &, TimePartitionLogger &);
  TimePartitionLogger *pLogger;
  friend class TimePartitionLogger;
};

/**
 * An extension to the Logger class which can summarize the times spent in
 * multiple mutually exclusive timing scopes.
 *
 * A TimePartitionLogger can be thought of as a set of stopwatches, where
 * there is never more than 1 stopwatch running at a time. Each stopwatch is
 * defined by a string.
 *
 * The TimePartitionLogger class might not work as expected with
 * multi-threadding.
 * */
class TimePartitionLogger : public Logger {

public:
  TimePartitionLogger(const std::string &id, bool extendIdToMakeUnique)
      : Logger(id, extendIdToMakeUnique),
        constructionTime(std::chrono::high_resolution_clock::now()) {}

  /**
   * Start the stopwatch \a stopwatch. The behaviour in the case where there
   * is already a stopwatch on depends on the inheriting class's
   * implementation of \a preHandleStartFromOn
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
   * A summary of the times on each stopwatch. An example might be:
   *
   *  first-stopwatch         : 0.005162 [s]    :    66 %
   *  second-stopwatch        : 0.002535 [s]    :    31 %
   *  unaccounted             : 0.000034 [s]    :     3 %
   *  total                   : 0.007730 [s]    :   100 %.
   * */
  std::string str(double minPercentage) const;

  /**
   * Print the summary string to std::cout, if this TimePartitionLogger's
   * Level is at least as high as \a l. See the base Logger class for
   * information on setting this TimePartitionLogger's Level.
   *
   * \sa str
   * */
  void summarize(double minPercentage, Level l) const;

  /**
   * Summarize, at Level Info.
   * */
  void summarizeInfo(double minPercentage) const;

  /**
   * Append summary string to output stream \a ost
   * \sa str
   * */
  void append(std::ostream &ost, double minPercentage) const;

  /**
   * Return the total time spent in stopwatch \a s, in seconds.
   * */
  double get(const std::string &s) const;

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

  /** \return true if there is currently a stopwatch which is on. */
  bool isOn() const { return clockState == ClockState::On; }

  /** \return true if currently all stopwatches are off. */
  bool isOff() const { return !isOn(); }

  /**
   * An Event: when a stopwatch either starts of stops.
   * */
  struct Event {
    std::string stopwatch;
    enum class Type { Start, Stop } type;
    double sinceConstruction;
  };

  /** \return A vector of all the Events registered. */
  const std::vector<Event> &events() const { return events_; }

  /** \return A string of all the Events registered. */
  std::string eventsStr() const;

  /**
   * For testing purposes: verify that all the Events registered match the
   * Events in \a expected, excluding the times of Events.
   */
  void verifyEvents(
      const std::vector<std::pair<std::string, Event::Type>> &expected) const;

  /**
   * How long has the current stopwatch been on for, if any stopwatches are
   * on.
   * */
  double beenOnFor() const;

  std::string currentStopwatch() const;

protected:
  void registerStartEvent(const std::string &stopwatch);
  void registerStopEvent();

private:
  // Cumulative times on each of the stopwatches.
  std::map<std::string, double> stopwatches;

  // Are there any stopwatches currently running?
  enum class ClockState { On, Off };
  ClockState clockState{ClockState::Off};

  std::string currentStopwatch_;

  // m[k] += t if k in m, else m[k] = t.
  void increment(std::map<std::string, double> &m,
                 const std::string &k,
                 double t) const;

  void increment(const std::string &k, double t) {
    increment(stopwatches, k, t);
  }

  void increment(double t) { increment(currentStopwatch(), t); }

  using TimePoint = std::chrono::high_resolution_clock::time_point;

  /**
   * Handle the case of starting a stopwatch, when there is already one on.
   */
  virtual void preHandleStartFromOn(const std::string &stopwatch) = 0;

  /**
   * Handle the case of stopping a stopwatch, when the stopwatch being
   * stopped was started when another stopwatch was on.
   */
  virtual void postHandleStartFromOn() = 0;

  TimePoint constructionTime;
  TimePoint lastEventTime;
  std::vector<Event> events_;

  std::mutex mut;
};

/**
 * Throw an error if a new stopwatch is started before stopping the current
 * one.
 * */
class ManualTimePartitionLogger : public TimePartitionLogger {
public:
  ManualTimePartitionLogger(const std::string &id, bool extendIdToMakeUnique)
      : TimePartitionLogger(id, extendIdToMakeUnique) {}

  explicit ManualTimePartitionLogger(const std::string &id)
      : ManualTimePartitionLogger(id, false) {}

  ManualTimePartitionLogger() : ManualTimePartitionLogger({}, true) {}

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
  SwitchingTimePartitionLogger(const std::string &id,
                               bool extendIdToMakeUnique)
      : TimePartitionLogger(id, extendIdToMakeUnique) {}

  explicit SwitchingTimePartitionLogger(const std::string &id)
      : SwitchingTimePartitionLogger(id, false) {}

  SwitchingTimePartitionLogger() : SwitchingTimePartitionLogger({}, true) {}

private:
  void preHandleStartFromOn(const std::string &stopwatch) final;
  void postHandleStartFromOn() final;

  std::vector<std::string> onHoldStack;
};

} // namespace logging
} // namespace poprithms

#endif
