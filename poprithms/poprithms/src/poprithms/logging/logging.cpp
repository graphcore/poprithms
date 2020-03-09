#ifndef USE_SPD_LOG
static_assert(
    false,
    "Currently, logging is only supported with SPDLOG, A non-SPDLOG version "
    "will needs to be implemented to mirror logging.cpp");
#endif

#include <experimental/propagate_const>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <poprithms/logging/logging.hpp>

namespace spdlog {
class logger;
}

namespace poprithms {
namespace logging {

namespace {
auto getSpdLogLevel(Level level) {

  switch (level) {
  case (Level::Trace): {
    return spdlog::level::trace;
  }
  case (Level::Debug): {
    return spdlog::level::debug;
  }
  case (Level::Info): {
    return spdlog::level::info;
  }
  case (Level::Off): {
    return spdlog::level::off;
  }
  }
}
} // namespace

void setGlobalLevel(Level l) { spdlog::set_level(getSpdLogLevel(l)); }

std::ostream &operator<<(std::ostream &os, Level l) {
  os << "Level::";
  switch (l) {
  case (Level::Trace): {
    os << "Trace";
    return os;
  }
  case (Level::Debug): {
    os << "Debug";
    return os;
  }
  case (Level::Info): {
    os << "Info";
    return os;
  }
  case (Level::Off): {
    os << "Off";
    return os;
  }
  }
}

class LoggerImpl {
public:
  LoggerImpl(const std::string &id) {
    lggr = spdlog::stdout_color_mt(id);
    // https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
    lggr->set_pattern("[%H:%M:%S.%e] [%n] [%^%l%$] %v");
  }
  std::experimental::propagate_const<std::shared_ptr<spdlog::logger>> lggr;
};

Logger::~Logger() = default;

// _mt : a thread safe logger (_mt : multi-threading)
// https://github.com/gabime/spdlog/wiki/1.1.-Thread-Safety
Logger::Logger(const std::string &id)
    : impl(std::make_unique<LoggerImpl>(id)) {}

void Logger::info(const std::string &s) { impl->lggr->info(s); }
void Logger::debug(const std::string &s) { impl->lggr->debug(s); }
void Logger::trace(const std::string &s) { impl->lggr->trace(s); }

void Logger::setLevel(Level level) {
  impl->lggr->set_level(getSpdLogLevel(level));
}

Level Logger::getLevel() const {
  switch (impl->lggr->level()) {
  case spdlog::level::trace: {
    return Level::Trace;
  }
  case spdlog::level::debug: {
    return Level::Debug;
  }
  case spdlog::level::info: {
    return Level::Info;
  }
  case spdlog::level::off: {
    return Level::Off;
  }

  // These cases should never happen, as poprithms has no way of setting these
  case spdlog::level::warn:
  case spdlog::level::err:
  case spdlog::level::critical:
  default: {
    throw std::runtime_error("Unsupported log level in poprithms");
  }
  }
}

bool Logger::shouldLog(Level l) const {
  return impl->lggr->should_log(getSpdLogLevel(l));
}

} // namespace logging
} // namespace poprithms
