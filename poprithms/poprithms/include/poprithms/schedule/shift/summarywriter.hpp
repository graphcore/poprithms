// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SUMMARYWRITER_HPP
#define POPRITHMS_SCHEDULE_SHIFT_SUMMARYWRITER_HPP

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/schedulechange.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

class ScheduledGraph;

/**
 * Abstract base class for writing summaries of a ScheduledGraph's input,
 * output, and performance. The virtual methods are called in specific places
 * while constructing a ScheduledGraph (construction, at completion, when the
 * schedule changes, etc.). By inheriting from this base class, it gives a
 * good amount of control of exactly what information is extracted or written
 * to file at the various steps.
 * */
class ISummaryWriter {
public:
  ISummaryWriter()          = default;
  virtual ~ISummaryWriter() = default;

  /** Is there a chance that the method 'write' will be called on the Graph
   * #initialGraph, to write it to cache once it has been constructed? */
  virtual bool mightWrite(const Graph &initialGraph) const = 0;

  /**
   * Is it certain that 'write' will be called on the Graph #initialGraph,
   * if it took #totalTime to construct it?
   * */
  virtual bool willWrite(const Graph &initialGraph,
                         double totalTime) const = 0;

  virtual void write(const Graph & /* initialGraph */,
                     const Graph & /* preShifting */,
                     double /* totalTime */,
                     const std::string & /* additional */) const = 0;

  /**
   * Every time a rotation is applied to the graph, this method is called.
   * */
  virtual void appendScheduleChange(const ScheduleChange &sc) const = 0;

  virtual void appendLivenessProfile(const ScheduledGraph &) const = 0;

  virtual void writeInitialSchedule(const std::vector<OpAddress> &) const = 0;

  virtual void writeFinalSchedule(const std::vector<OpAddress> &) const = 0;
};

class FileWriter : public ISummaryWriter {
public:
  // Checks environment variables, otherwise retuns None.
  static FileWriter Default();

  // Never writes, even if environment variables are set.
  static FileWriter None() { return FileWriter({}, 0); }

  void appendScheduleChange(const ScheduleChange &) const final {}

  void appendLivenessProfile(const ScheduledGraph &) const final {}

  void writeInitialSchedule(const std::vector<OpAddress> &) const final {}

  void writeFinalSchedule(const std::vector<OpAddress> &) const final {}

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
  FileWriter(const std::string &baseDirectory,
             uint64_t maxWritesPerBin = defaultMaxWritesPerBin());

  static uint64_t defaultMaxWritesPerBin() { return 2; }

  /**
   * \param fromUser This should be the Graph that the user passes to the
   *              ScheduledGraph constructor.
   *
   * \param preShifting This should be the Graph whose schedule is optimized
   * for after the initial transitive closure passes.
   *
   * \param additional This should be a summary of the time spent in top-level
   *                   scheduling algorithms.
   * */
  void write(const Graph &fromUser,
             const Graph &preShifting,
             double totalTime,
             const std::string &additional) const final;

  bool mightWrite(const Graph & /* fromUser */) const final {
    return maxWritesPerBin > 0;
  }

  bool willWrite(const Graph &, /* fromUser */
                 double /* totalTime */) const final;

  /**
   * You can export an environment variable to set the base directory for
   * writing summaries to:
   *
   * >> export POPRITHMS_SCHEDULE_SHIFT_WRITE_DIRECTORY=/path/to/write/dir
   *
   * this environment variable will be used if the empty-string is passed to
   * the constructor. If a FileWriter is constructed with a
   * non-empty string, then the environment variable is ignored.
   *
   * */
  static constexpr const char *const dirEnv =
      "POPRITHMS_SCHEDULE_SHIFT_WRITE_DIRECTORY";

  static constexpr const char *const maxWritesPerBinEnv =
      "POPRITHMS_SCHEDULE_SHIFT_MAX_WRITES_PER_BIN";

  static std::string
  finalDirName(uint64_t tSeconds, uint64_t nOps, uint64_t uid);

private:
  std::string dir_;
  uint32_t maxWritesPerBin;

  static std::mutex mut;
  uint64_t getUid(uint64_t tSeconds, uint64_t nOps) const;
  std::string dirName(uint64_t tSeconds, uint64_t nOps, uint64_t uid) const;
};

class SwitchSummaryWriter : public ISummaryWriter {

public:
  SwitchSummaryWriter();

  bool mightWrite(const Graph & /* fromUser */) const final { return true; }

  bool willWrite(const Graph &, /* fromUser */
                 double /* totalTime */) const final {
    return true;
  }

  void write(const Graph & /* fromUser */,
             const Graph & /* preShifting */,
             double /* totalTime */,
             const std::string & /* additional */) const final;

  void appendScheduleChange(const ScheduleChange &sc) const final {
    allInfo->allChanges.push_back(sc);
  }

  void appendLivenessProfile(const ScheduledGraph &) const final;

  void writeInitialSchedule(const std::vector<OpAddress> &is) const final {
    allInfo->initialSchedule = is;
  }

  void writeFinalSchedule(const std::vector<OpAddress> &is) const final {
    allInfo->finalSchedule = is;
  }

  void writeToFile(const std::string &dirName = {}) const {
    allInfo->writeToFile(dirName);
  }

  const std::vector<ScheduleChange> &allChanges() const {
    return allInfo->allChanges;
  }

private:
  struct AllInfo {
    Graph fromUser;
    Graph preShifting;
    std::vector<OpAddress> initialSchedule;
    std::vector<OpAddress> finalSchedule;
    std::vector<ScheduleChange> allChanges;
    std::vector<std::vector<AllocWeight>> livenessProfiles;
    void writeToFile(const std::string &dirName) const;

    std::string fromUserFn{"graphFromUser.json"};
    std::string preShiftingFn{"graphPreShifting.json"};
    std::string initialScheduleFn{"initialSchedule.txt"};
    std::string finalScheduleFn{"finalSchedule.txt"};
    std::string shiftsFn{"shifts.txt"};
    std::string livenessProfilesFn{"livenessProfiles.txt"};
  };

  std::unique_ptr<AllInfo> allInfo;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
