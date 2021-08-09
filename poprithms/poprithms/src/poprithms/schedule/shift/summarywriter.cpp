// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cctype>
#include <schedule/shift/error.hpp>

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/format/format_fwd.hpp>

#include <poprithms/schedule/shift/summarywriter.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

bool SummaryWriter::isWhitespace(const std::string &s) {
  return std::all_of(
      s.cbegin(), s.cend(), [](auto c) { return std::isspace(c); });
}

void SummaryWriter::write(const Graph &start,
                          const Graph &end,
                          double total,
                          const std::string &additional) const {

  using boost::filesystem::path;

  // directory names are
  //
  // time<number of seconds>__
  // nOps<number of ops in start graph>__
  // uid<a number to make the directory name unique>
  auto subDir = [this, total, &start]() {
    uint64_t totalSeconds = static_cast<uint64_t>(total);
    uint64_t uid{0};
    auto nxt = [this, totalSeconds, &uid, &start]() {
      std::ostringstream oss;
      oss << "time" << totalSeconds << "__"
          << "nOps" << start.nOps() << "__uid" << uid;
      return path(dir_) / oss.str();
    };
    while (boost::filesystem::exists(nxt())) {
      uid += 1;
    }
    return nxt();
  }();

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
}

SummaryWriter::SummaryWriter(const std::string &bd) : dir_(bd) {

  // The user can do
  // >> export POPRITHMS_SCHEDULE_SHIFT_WRITE_DIRECTORY=/path/to/write/dir
  //
  // which will be used if the string passed to the constructor is empty.

  if (isWhitespace(bd)) {

    // I could canonicalize the path with boost::canonicalize, but that might
    // cause confusion about what the the path is relative to. So i'm leaving
    // the responsibility up to the user to provide absolute paths.
    // TLDR; ~/some/path is fine but ../../some/path might not be.
    const char *const fromEnvVar = std::getenv(dirEnv);
    if (fromEnvVar) {
      dir_ = std::string(fromEnvVar);
      if (!dir_.empty()) {
        dirFromEnvVariable_ = true;
      }
    }
  }

  if (!isWhitespace(dir_)) {
    if (!boost::filesystem::exists(dir_)) {
      std::ostringstream oss;
      oss << "The directory '" + dir_ + "' does not exist. ";
      oss << "This directory name was ";
      if (dirFromEnvVariable_) {
        oss << "obtained from the environment variable '" << dirEnv << "'. "
            << "Either set this variable to a valid directory name, or unset "
               "it. ";

      } else {
        oss << "provided to the SummaryWriter constructor.";
      }
      throw error(error::Code(12345), oss.str());
    }
  }
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
