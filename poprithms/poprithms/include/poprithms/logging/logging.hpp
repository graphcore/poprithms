// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_LOGGING_LOGGING_HPP
#define POPRITHMS_LOGGING_LOGGING_HPP

#include <memory>
#include <string>

namespace poprithms {
namespace logging {

class LoggerImpl;

enum class Level { Trace = 0, Debug, Info, Off, NumberOfLevels };
std::ostream &operator<<(std::ostream &, Level);
Level getLevel(const std::string &);
std::string getName(Level);

// Set the logging level for all Loggers. Example:
//
//                                A     B     C     D
// Logger A("a");                Off   --    --    --
// Logger B("b");                Off   Off   --    --
// setGlobalLevel(Level::Info);  Info  Info  --    --
// Logger C("c");                Info  Info  Info  --
// setGlobalLevel(Level::Debug); Debug Debug Debug --
// B.setLevel(Level::Off);       Debug Off   Debug --
// setGlobalLevel(Level::Info);  Info  Info  Info  --
// Logger D("d");                Info  Info  Info  Info
// A.setLevel(Level::Off);       Off   Info  Info  Info
// Logger E("a");                Error: cannot have 2 Loggers with same name
//
//

void setGlobalLevel(Level);

// By default there is no timing information with logging. It can be enabled
// with these functions
//
// Log the time taken between successive log lines
void enableDeltaTime(bool);
// Log the total time taken since execution commenced
void enableTotalTime(bool);

class Logger {
public:
  /**
   * Create a Logger with a unique id.
   *
   * \param id The id of the Logger, must be unique. If another Logger exists
   *           with this id, an error is thrown.
   * */
  explicit Logger(const std::string &id) : Logger(id, false) {}

  /**
   * Create a Logger from a proposed id, with the option to extend the
   * proposed id with a random character string if there is already Logger
   * with the id \a id.
   *
   * \param id The proposed id for the Logger
   *
   * \param extendIdToMakeUnique if false, and there is already a Logger with
   *                             id \a id, then throw an error. if true, and
   *                             there is already a Logger with id \a id, then
   *                             try adding a random character string to \a id
   *                             to fond a unique id. Try this until a unique
   *                             id is found.
   * */
  explicit Logger(const std::string &id, bool extendIdToMakeUnique);

  /**
   * Create a Logger with a unique name of random characters.
   * */
  Logger() : Logger({}, true){};

  virtual ~Logger();
  void info(const std::string &) const;
  void debug(const std::string &) const;
  void trace(const std::string &) const;

  void setLevel(Level);
  void setLevelInfo() { setLevel(Level::Info); }
  void setLevelDebug() { setLevel(Level::Debug); }
  void setLevelTrace() { setLevel(Level::Trace); }
  void setLevelOff() { setLevel(Level::Off); }

  Level getLevel() const;

  //
  // In the following table, at "x" logging will be produced
  //
  //                                 getLevel()
  //                        -----------------------
  //                        Trace  Debug  Info  Off
  //
  //            Trace         x      .     .     .
  // atLevel    Debug         x      x     .     .
  //            Info          x      x     x     .
  //
  bool shouldLog(Level atLevel) const;
  bool shouldLogInfo() const { return shouldLog(Level::Info); }
  bool shouldLogDebug() const { return shouldLog(Level::Debug); }
  bool shouldLogTrace() const { return shouldLog(Level::Trace); }

  /**
   * Return the unique identifier of this Logger. This is the same string as
   * was passed into the constructor.
   * */
  std::string id() const;

private:
  LoggerImpl *impl;
};

} // namespace logging
} // namespace poprithms

#endif
