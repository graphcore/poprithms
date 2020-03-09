#ifndef POPRITHMS_LOGGING_LOGGING_HPP
#define POPRITHMS_LOGGING_LOGGING_HPP

#include <experimental/propagate_const>
#include <memory>
#include <string>

namespace poprithms {
namespace logging {

class LoggerImpl;

enum class Level { Trace = 0, Debug, Info, Off };
std::ostream &operator<<(std::ostream &, Level);

// Set the logging level for all Loggers
void setGlobalLevel(Level);

class Logger {
public:
  Logger(const std::string &id);
  ~Logger();
  void info(const std::string &);
  void debug(const std::string &);
  void trace(const std::string &);

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

private:
  // we use propagate_const to be stricter than just bitwise constness
  // https://en.cppreference.com/w/cpp/experimental/propagate_const
  std::experimental::propagate_const<std::unique_ptr<LoggerImpl>> impl;
};

} // namespace logging
} // namespace poprithms

#endif
