// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SUMMARYWRITER_HPP
#define POPRITHMS_SCHEDULE_SHIFT_SUMMARYWRITER_HPP

#include <mutex>
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
};

class SummaryWriter : public ISummaryWriter {
public:
  // Checks environment variables, otherwise retuns None.
  static SummaryWriter Default();

  // Never writes, even if environment variables are set.
  static SummaryWriter None() { return SummaryWriter({}, 0); }

  /**
   * \param baseDirectory the base directory to which summary files will be
   *                      written. A subdirectory of this base directory will
   *                      be created for the files. The subdirectory will be
   *                      based on (1) the number of Ops in the Graph and (2)
   *                      the total time (nearest second) it took to schedule
   *                      the Graph.
   *
   * \param maxWritesPerBin A bin is defined by (1) and (2), see above. This
   *                        argument controls the number of new directories
   *                        which are created.
   * */
  SummaryWriter(const std::string &baseDirectory,
                uint64_t maxWritesPerBin = defaultMaxWritesPerBin());

  static uint64_t defaultMaxWritesPerBin() { return 2; }

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

  /**
   * You can export an environment variable to set the base directory for
   * writing summaries to:
   *
   * >> export POPRITHMS_SCHEDULE_SHIFT_WRITE_DIRECTORY=/path/to/write/dir
   *
   * this environment variable will be used if the empty-string is passed to
   * the constructor. If a SummaryWriter is constructed with a non-empty
   * string, then the environment variable is ignored.
   *
   * */
  static constexpr const char *const dirEnv =
      "POPRITHMS_SCHEDULE_SHIFT_WRITE_DIRECTORY";

  static constexpr const char *const maxWritesPerBinEnv =
      "POPRITHMS_SCHEDULE_SHIFT_MAX_WRITES_PER_BIN";

private:
  std::string dir_;
  uint32_t maxWritesPerBin;

  static std::mutex mut;
  uint64_t getUid(uint64_t tSeconds, uint64_t nOps) const;
  std::string dirName(uint64_t tSeconds, uint64_t nOps, uint64_t uid) const;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
