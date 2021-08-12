// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cctype>
#include <mutex>
#include <schedule/shift/error.hpp>

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/format/format_fwd.hpp>

#include <poprithms/schedule/shift/summarywriter.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

std::string SummaryWriter::dirName(uint64_t totalSeconds,
                                   uint64_t nOps,
                                   uint64_t uid) const {
  std::ostringstream oss;
  oss << "time" << totalSeconds << "__"
      << "nOps" << nOps << "__uid" << uid;
  auto nxt = boost::filesystem::path(dir_) / oss.str();
  return nxt.string();
}

uint64_t SummaryWriter::getUid(uint64_t totalSeconds, uint64_t nOps) const {
  uint64_t uid{0};
  bool exists{true};
  while (exists) {
    exists = boost::filesystem::exists(dirName(totalSeconds, nOps, uid));
    if (exists) {
      ++uid;
    }
  }
  return uid;
}

std::mutex SummaryWriter::mut;

void SummaryWriter::write(const Graph &start,
                          const Graph &end,
                          double total,
                          const std::string &additional) const {

  std::lock_guard<std::mutex> goo(mut);

  auto uid = getUid(static_cast<uint64_t>(total), start.nOps());
  if (uid >= maxWritesPerBin) {
    return;
  }

  auto dn     = dirName(static_cast<uint64_t>(total), start.nOps(), uid);
  auto subDir = boost::filesystem::path(dn);
  boost::filesystem::create_directory(subDir);

  {
    std::ofstream out((subDir / "graph0.json").string());
    out << start.getSerializationString();
    out.close();
  }

  {
    std::ofstream out((subDir / "graph1.json").string());
    out << end.getSerializationString();
    out.close();
  }

  {
    std::ofstream out((subDir / "summary.json").string());
    out << additional;
    out.close();
  }

  {
    std::ofstream out((subDir / "dag1.txt").string());
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

SummaryWriter::SummaryWriter(const std::string &bd, uint64_t maxWritesPerBin_)
    : dir_(bd), maxWritesPerBin(maxWritesPerBin_) {

  if (maxWritesPerBin_ > 0 && !boost::filesystem::exists(dir_)) {
    std::ostringstream oss;
    oss << "The directory '" + dir_ + "' does not exist. ";
    oss << "This directory name was ";
    oss << "provided to the SummaryWriter constructor, along with "
        << "maxWritesPerBin=" << maxWritesPerBin_ << '.';

    throw error(error::Code(12345), oss.str());
  }
}

SummaryWriter SummaryWriter::Default() {

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
    return SummaryWriter(dir, defaultMaxWritesPerBin());
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

  return SummaryWriter(dir, std::stoi(maxCountsString));
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
