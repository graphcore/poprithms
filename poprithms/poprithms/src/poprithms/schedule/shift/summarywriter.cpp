// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cctype>
#include <mutex>

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/format/format_fwd.hpp>

#include <schedule/shift/error.hpp>

#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <poprithms/schedule/shift/summarywriter.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

namespace {
bool dirExists(const std::string &dn) {
  return boost::filesystem::exists(dn);
}

std::string getFileName(const std::string &dn, const std::string &fn) {
  auto nxt = boost::filesystem::path(dn) / fn;
  return nxt.string();
}

void createDirectory(const std::string &dn) {
  auto subDir = boost::filesystem::path(dn);
  boost::filesystem::create_directory(subDir);
}
} // namespace

std::string
FileWriter::finalDirName(uint64_t totalSeconds, uint64_t nOps, uint64_t uid) {
  std::ostringstream oss;
  oss << "time" << totalSeconds << "__"
      << "nOps" << nOps << "__uid" << uid;
  return oss.str();
}

std::string
FileWriter::dirName(uint64_t tSeconds, uint64_t nOps, uint64_t uid) const {
  return getFileName(dir_, finalDirName(tSeconds, nOps, uid));
}

uint64_t FileWriter::getUid(uint64_t totalSeconds, uint64_t nOps) const {
  uint64_t uid{0};
  bool exists{true};
  while (exists) {
    exists = dirExists(dirName(totalSeconds, nOps, uid));
    if (exists) {
      ++uid;
    }
  }
  return uid;
}

std::mutex FileWriter::mut;

bool FileWriter::willWrite(const Graph &start, double totalTime) const {

  std::lock_guard<std::mutex> goo(mut);

  const auto timeSeconds = static_cast<uint64_t>(totalTime);
  const auto uid         = getUid(timeSeconds, start.nOps());
  return uid < maxWritesPerBin;
}

void SwitchSummaryWriter::AllInfo::writeToFile(
    const std::string &dirName) const {

  createDirectory(dirName);

  {
    std::ofstream out(getFileName(dirName, fromUserFn));
    out << fromUser.getSerializationString();
    out.close();
  }

  {
    std::ofstream out(getFileName(dirName, preShiftingFn));
    out << preShifting.getSerializationString();
    out.close();
  }

  {
    std::ofstream out(getFileName(dirName, initialScheduleFn));
    poprithms::util::append(out, initialSchedule);
    out.close();
  }

  {
    std::ofstream out(getFileName(dirName, finalScheduleFn));
    poprithms::util::append(out, finalSchedule);
    out.close();
  }

  {
    std::ofstream out(getFileName(dirName, livenessProfilesFn));
    for (const auto &lp : livenessProfiles) {
      poprithms::util::append(out, lp);
      out << '\n';
    }
    out.close();
  }

  {
    std::ofstream out(getFileName(dirName, shiftsFn));
    for (auto x : allChanges) {
      out << x << '\n';
    }
    out.close();
  }
}

void FileWriter::write(const Graph &start,
                       const Graph &end,
                       double totalTime,
                       const std::string &additional) const {

  std::lock_guard<std::mutex> goo(mut);

  const auto timeSeconds = static_cast<uint64_t>(totalTime);
  auto uid               = getUid(timeSeconds, start.nOps());

  const auto subDir = dirName(timeSeconds, start.nOps(), uid);
  createDirectory(subDir);

  {
    std::ofstream out(getFileName(subDir, "graph0.json"));
    out << start.getSerializationString();
    out.close();
  }

  {
    std::ofstream out(getFileName(subDir, "graph1.json"));
    out << end.getSerializationString();
    out.close();
  }

  {

    std::ofstream out(getFileName(subDir, "summary.txt"));
    out << additional;
    out.close();
  }

  {
    std::ofstream out(getFileName(subDir, "dag1.txt"));
    std::ostringstream oss;
    auto edges = end.getForwardEdges();
    for (uint64_t i = 0; i < edges.size(); ++i) {
      oss << i << ":";
      poprithms::util::append(oss, edges[i]);
      oss << '\n';
    }
    out << oss.str();
    out.close();
  }
}

FileWriter::FileWriter(const std::string &bd, uint64_t maxWritesPerBin_)
    : dir_(bd), maxWritesPerBin(maxWritesPerBin_) {

  if (maxWritesPerBin_ > 0 && !boost::filesystem::exists(dir_)) {
    std::ostringstream oss;
    oss << "The directory '" + dir_ + "' does not exist. ";
    oss << "This directory name was ";
    oss << "provided to the FileWriter constructor, along with "
        << "maxWritesPerBin=" << maxWritesPerBin_ << '.';

    throw error(error::Code(12345), oss.str());
  }
}

FileWriter FileWriter::Default() {

  const char *const fromEnvVar = std::getenv(dirEnv);
  if (!fromEnvVar) {
    return None();
  }
  auto dir = std::string(fromEnvVar);

  if (!boost::filesystem::exists(dir)) {
    std::ostringstream oss;
    oss << "The directory '" + dir + "' does not exist. ";
    oss << "This directory name was ";
    oss << "obtained from the environment variable '" << dirEnv << "'. "
        << "Either set this variable to a valid directory name, or unset "
           "it. ";
    throw error(oss.str());
  }

  const char *const maxCounts = std::getenv(maxWritesPerBinEnv);
  if (!maxCounts) {
    return FileWriter(dir, defaultMaxWritesPerBin());
  }

  const auto maxCountsString = std::string(maxCounts);

  // check if it can be cast to an unsigned integer.
  if (maxCountsString.empty() ||
      maxCountsString.find_first_not_of("0123456789") != std::string::npos) {
    std::ostringstream oss;
    oss << "Invalid environment variable " << maxWritesPerBinEnv << ", '"
        << maxCounts << "'. It must be non-empty and contain only digits. ";
    throw error(oss.str());
  }

  return FileWriter(dir, std::stoi(maxCountsString));
}

void SwitchSummaryWriter::write(const Graph &fu,
                                const Graph &ps,
                                double totalTime,
                                const std::string &additional) const {
  allInfo->fromUser    = fu;
  allInfo->preShifting = ps;
  (void)totalTime;
  (void)additional;
}

void SwitchSummaryWriter::appendLivenessProfile(
    const ScheduledGraph &sg) const {
  auto liveness = sg.getSchToLiveness();
  allInfo->livenessProfiles.push_back(liveness);
}

SwitchSummaryWriter::SwitchSummaryWriter()
    : allInfo(std::make_unique<AllInfo>()) {}

void ISummaryWriter::noWeakVTables() {
  throw error(error::error::weakVTableMessage());
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
