// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <schedule/shift/error.hpp>

#include <poprithms/schedule/shift/settings.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

RotationTermination Settings::defaultRotationTermination() {
  return {defaultRotationLimitSeconds(), defaultRotationLimitCount()};
}

KahnTieBreaker Settings::defaultKahnTieBreaker() {
  return KahnTieBreaker::GREEDY;
}

TransitiveClosureOptimizations Settings::defaultTCOs() {
  // TODO(T19732) change to allOn(). Make sure all buildbots are happy with
  // this before landing.
  return TransitiveClosureOptimizations::allOff();
}

std::ostream &operator<<(std::ostream &ost, const DebugMode &dbm) {

  switch (dbm) {
  case (DebugMode::On): {
    ost << "DebugMode::On";
    break;
  }
  case (DebugMode::Off): {
    ost << "DebugMode::Off";
    break;
  }
  }

  return ost;
}

Settings::Settings(const std::map<std::string, std::string> &m) : Settings() {

  for (const auto &[k, v] : m) {
    if (k == "allTCO") {
      const auto allTCOs = static_cast<bool>(std::stoi(v));
      tcos_              = allTCOs ? TransitiveClosureOptimizations::allOn()
                      : TransitiveClosureOptimizations::allOff();
    }

    else if (k == "seed") {
      seed_ = static_cast<uint32_t>(std::stoul(v));
    }

    else if (k == "tieBreaker" || k == "kahnTieBreaker") {
      kd_ = {shift::kahnTieBreaker(v), {}};
    }

    else if (k == "debug") {
      dm_ = static_cast<bool>(std::stoi(v)) ? DebugMode::On : DebugMode::Off;
    }

    else if (k == "timeLimitSeconds") {
      rt_.setMaxSeconds(static_cast<double>(std::stod(v)));
    }

    else if (k == "swapLimitCount" || k == "rotationCountLimit") {
      rt_.setMaxRotations(static_cast<int64_t>(std::stoll(v)));
    }

    else {
      throw error("invalid option in greedyRotate, " + k);
    }
  }
}
} // namespace shift
} // namespace schedule
} // namespace poprithms
