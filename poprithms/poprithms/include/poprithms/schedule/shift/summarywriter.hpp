// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SUMMARYWRITER_HPP
#define POPRITHMS_SCHEDULE_SHIFT_SUMMARYWRITER_HPP

#include <string>
#include <vector>

#include <poprithms/schedule/shift/graph.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

/**
 * Abstract base class for writing summaries of a ScheduledGraph's input,
 * output, and performance.
 * */
class ISummaryWriter {
public:
  virtual void write(const Graph &start,
                     const Graph &end,
                     double total,
                     const std::string &additional) const = 0;

  /**
   * return true if no summary should be written.
   * */
  virtual bool empty() const = 0;
};

class SummaryWriter : public ISummaryWriter {
public:
  static SummaryWriter None() { return SummaryWriter({}); }

  /**
   * \param baseDirectory the directory to which summary files will be
   *                      written.
   * */
  SummaryWriter(const std::string &baseDirectory);

  /**
   * \param start This should be the Graph that the user passes to the
   *              ScheduledGraph constructor.
   *
   * \param end This should be the Graph whose schedule is optimized for after
   *            the initial transitive closure passes.
   *
   * \param total This should be the total time spent obtaining a schedule.
   *
   * \param additional This should be a summary of the time spent in top-level
   *                   scheduling algorithms.
   * */
  void write(const Graph &start,
             const Graph &end,
             double total,
             const std::string &additional) const final;

  bool empty() const final { return isWhitespace(dir_); }

  static constexpr const char *const dirEnv =
      "POPRITHMS_SCHEDULE_SHIFT_WRITE_DIRECTORY";

private:
  std::string dir_;

  static bool isWhitespace(const std::string &);

  /**
   * The user can export an environment variable to set the base directory for
   * writing summaries to:
   *
   * >> export POPRITHMS_SCHEDULE_SHIFT_WRITE_DIRECTORY=/path/to/write/dir
   *
   * this environment variable will be used if the empty-string is passed to
   * the constructor. If a SummaryWriter is constructed with a non-empty
   * string, then the environment variable is ignored.
   *
   * This bool records if the directory was set by the environment variable.
   * */
  bool dirFromEnvVariable_{false};
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
