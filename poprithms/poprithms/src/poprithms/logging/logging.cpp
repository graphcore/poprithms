#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <poprithms/logging/error.hpp>
#include <poprithms/logging/logging.hpp>

namespace poprithms {
namespace logging {

class LoggerImpl {
public:
  LoggerImpl(const std::string &_id_);
  Level getLevel() const;
  void setLevel(Level _l_);
  const std::string &getId();

private:
  std::string id;
  Level level;
};

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

namespace {

class LoggerImplContainer {

public:
  LoggerImpl *getLoggerImpl(const std::string &id) {
    auto found = impls.find(id);
    if (found != impls.cend()) {
      throw logging::error("There is already a Logger with id `" + id + "'.");
    }
    impls[id] = std::make_unique<LoggerImpl>(id);
    return impls[id].get();
  }

  void setGlobalLevel(Level l) {
    globalLevel = l;
    for (auto &x : impls) {
      x.second->setLevel(globalLevel);
    }
  }

  Level getGlobalLevel() const { return globalLevel; }

private:
  std::map<std::string, std::unique_ptr<LoggerImpl>> impls;
  Level globalLevel{Level::Off};
} implContainer;

} // namespace

LoggerImpl::LoggerImpl(const std::string &_id_)
    : id(_id_), level(implContainer.getGlobalLevel()) {}

Level LoggerImpl::getLevel() const { return level; }

void LoggerImpl::setLevel(Level _l_) { level = _l_; }

const std::string &LoggerImpl::getId() { return id; }

void setGlobalLevel(Level l) { implContainer.setGlobalLevel(l); }

void Logger::info(const std::string &x) const {
  if (shouldLog(Level::Info)) {
    std::cout << '[' << impl->getId() << "] [info]  " << x << '\n';
  }
}

void Logger::debug(const std::string &x) const {
  if (shouldLog(Level::Debug)) {
    std::cout << '[' << impl->getId() << "] [debug] " << x << '\n';
  }
}

void Logger::trace(const std::string &x) const {
  if (shouldLog(Level::Trace)) {
    std::cout << '[' << impl->getId() << "] [trace] " << x << '\n';
  }
}

void Logger::setLevel(Level l) { impl->setLevel(l); }

Level Logger::getLevel() const { return impl->getLevel(); }

bool Logger::shouldLog(Level atLevel) const {

  auto current = getLevel();
  return static_cast<int>(current) <= static_cast<int>(atLevel);
}

Logger::~Logger() = default;

Logger::Logger(const std::string &id) {
  impl = implContainer.getLoggerImpl(id);
}

} // namespace logging
} // namespace poprithms
