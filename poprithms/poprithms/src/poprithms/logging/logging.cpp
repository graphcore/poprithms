#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>

#include <poprithms/logging/error.hpp>
#include <poprithms/logging/logging.hpp>

namespace poprithms {
namespace logging {

namespace {
class LoggerImplContainer;
}

class LoggerImpl {
public:
  LoggerImpl(const std::string &_id_, LoggerImplContainer *container);
  Level getLevel() const;
  void setLevel(Level _l_);
  const std::string &getId();
  std::string prefix();

private:
  std::string id;
  Level level;
  LoggerImplContainer *container;
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
  LoggerImplContainer() {
    tA = std::chrono::high_resolution_clock::now();
    tY = std::chrono::high_resolution_clock::now();
    tZ = std::chrono::high_resolution_clock::now();
  }

  LoggerImpl *getLoggerImpl(const std::string &id) {
    auto found = impls.find(id);
    if (found != impls.cend()) {
      throw logging::error("There is already a Logger with id `" + id + "'.");
    }
    impls[id] = std::make_unique<LoggerImpl>(id, this);
    return impls[id].get();
  }

  void setGlobalLevel(Level l) {
    globalLevel = l;
    for (auto &x : impls) {
      x.second->setLevel(globalLevel);
    }
  }

  void updateTime() {
    tY   = tZ;
    tZ   = std::chrono::high_resolution_clock::now();
    AtoZ = tZ - tA;
    YtoZ = tZ - tY;
  }

  std::string deltaTimeStr() const { return std::to_string(YtoZ.count()); }

  std::string totalTimeStr() const { return std::to_string(AtoZ.count()); }

  Level getGlobalLevel() const { return globalLevel; }

  void setDeltaTime(bool b) { deltaTime = b; }
  void setTotalTime(bool b) { totalTime = b; }

  bool logDeltaTime() const { return deltaTime; }
  bool logTotalTime() const { return totalTime; }

private:
  std::map<std::string, std::unique_ptr<LoggerImpl>> impls;
  Level globalLevel{Level::Off};

  using TimePoint    = decltype(std::chrono::high_resolution_clock::now());
  using TimeInterval = std::chrono::duration<double>;
  TimePoint tA;
  TimePoint tY;
  TimePoint tZ;
  TimeInterval AtoZ;
  TimeInterval YtoZ;

  bool deltaTime{false};
  bool totalTime{false};

} implContainer;

} // namespace

LoggerImpl::LoggerImpl(const std::string &_id_, LoggerImplContainer *c)
    : id(_id_), level(implContainer.getGlobalLevel()), container(c) {}

Level LoggerImpl::getLevel() const { return level; }

void LoggerImpl::setLevel(Level _l_) { level = _l_; }

const std::string &LoggerImpl::getId() { return id; }

void setGlobalLevel(Level l) { implContainer.setGlobalLevel(l); }

void enableDeltaTime(bool b) { implContainer.setDeltaTime(b); }
void enableTotalTime(bool b) { implContainer.setTotalTime(b); }

std::string LoggerImpl::prefix() {
  container->updateTime();
  std::ostringstream oss;
  if (container->logTotalTime()) {
    oss << "[T=" << container->totalTimeStr() << "] ";
  }
  if (container->logDeltaTime()) {
    oss << "[dt=" << container->deltaTimeStr() << "] ";
  }
  oss << "[" << getId() << ']';
  return oss.str();
}

void Logger::info(const std::string &x) const {
  if (shouldLog(Level::Info)) {
    std::cout << impl->prefix() << " [info]  " << x << std::endl;
  }
}

void Logger::debug(const std::string &x) const {
  if (shouldLog(Level::Debug)) {
    std::cout << impl->prefix() << " [debug] " << x << std::endl;
  }
}

void Logger::trace(const std::string &x) const {
  if (shouldLog(Level::Trace)) {
    std::cout << impl->prefix() << " [trace] " << x << std::endl;
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
