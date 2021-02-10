// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_LOGGING_SUMMARIZER_HPP
#define POPRITHMS_LOGGING_SUMMARIZER_HPP

#include <chrono>
#include <map>

#include <poprithms/logging/logging.hpp>

namespace poprithms {
namespace logging {

/**
 * An extension to the Logger class, which can summarize the times spent in
 * multiple mutually exclusive timing scopes.
 *
 * A TimeInScopesLogger can be thought of as a set of stopwatches,
 * where there is never more than 1 stopwatch running at a time. Each
 * stopwatch is defined by a string.
 *
 * The TimeInScopesLogger class might not work as expected with
 * multi-threadeding.
 * */
class TimeInScopesLogger : public Logger {

public:
  explicit TimeInScopesLogger(const std::string &id)
      : Logger(id),
        constructionTime(std::chrono::high_resolution_clock::now()) {}

  /**
   * Start, or restart timing the stopwatch defined by string \a stopwatch.
   *
   * There must be at least one call to stop (see below) between any 2 start
   * calls on different stopwatches.
   * */
  void start(const std::string &stopwatch);

  /** Stop the stopwatch that is currently running. If no stopwatches are
   * currently running, then do nothing. */
  void stop();

  /**
   * A summary of the times on each stopwatch. An example might be:
   *
   *  first-stopwatch         : 0.005162 [s]    :    66 %
   *  un-autre-chronometre    : 0.002535 [s]    :    32 %
   *  unaccounted             : 0.000034 [s]    :     0 %
   *  total                   : 0.007730 [s]    :   100 %.
   *
   *
   * */
  std::string str() const;

  /**
   * Print summary string to std::cout, if this TimeInScopesLogger's Level is
   * at least as high as \a l. See the base Logger class for information on
   * setting this TimeInScopesLogger's Level.
   *
   * \sa str
   * */
  void summarize(Level l = Level::Info) const;

  /**
   * Append summary string to output stream \a ost
   * \sa str
   * */
  void append(std::ostream &ost) const;

  /**
   * Return the total time spent in stopwatch \a s, in seconds.
   * */
  double get(const std::string &s) const;

  /**
   * Return the total elapsed time since this TimeInScopesLogger was
   * constructed, in seconds.
   * */
  double sinceConstruction() const;

  /**
   * Return the cumulative time on all stopwatches, in seconds.
   * */
  double accounted() const;

  double unaccounted() const { return sinceConstruction() - accounted(); }

private:
  double sinceCurrentStart() const;

  // Cumulative times on each of the stopwatches.
  std::map<std::string, double> stopwatches;

  // Are there any stopwatches currently running?
  enum class ClockState { On, Off };
  ClockState clockState{ClockState::Off};

  bool on() const { return clockState == ClockState::On; }
  bool off() const { return !on(); }

  void increment(std::map<std::string, double> &,
                 const std::string &k,
                 double t) const;

  // What is the currently running stopwatch?
  std::string currentStopwatch{"none"};

  using TimePoint = decltype(std::chrono::high_resolution_clock::now());
  TimePoint startOfCurrentStopwatch;
  TimePoint constructionTime;
};

std::ostream &operator<<(std::ostream &, const TimeInScopesLogger &);

} // namespace logging
} // namespace poprithms

#endif
