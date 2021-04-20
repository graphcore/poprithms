// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_SETTINGS
#define POPRITHMS_SCHEDULE_SHIFT_SETTINGS

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/kahntiebreaker.hpp>
#include <poprithms/schedule/shift/rotationalgo.hpp>
#include <poprithms/schedule/shift/rotationtermination.hpp>
#include <poprithms/schedule/shift/transitiveclosureoptimizations.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

enum class DebugMode { Off = 0, On };
std::ostream &operator<<(std::ostream &ost, const DebugMode &);

// How to schedule.
class Settings {

public:
  Settings(const std::map<std::string, std::string> &);
  Settings(KahnTieBreaker ktb                 = defaultKahnTieBreaker(),
           TransitiveClosureOptimizations tco = defaultTCOs(),
           RotationTermination rt             = defaultRotationTermination(),
           RotationAlgo algo                  = defaultRotationAlgo(),
           uint32_t seed                      = defaultSeed(),
           DebugMode dm                       = defaultDebugMode())
      : ktb_(ktb), tcos_(tco), rt_(rt), ra_(algo), seed_(seed), dm_(dm) {}

  KahnTieBreaker kahnTieBreaker() const { return ktb_; }
  TransitiveClosureOptimizations tcos() const { return tcos_; }
  RotationTermination rotationTermination() const { return rt_; }
  RotationAlgo rotationAlgo() const { return ra_; }
  uint32_t seed() const { return seed_; }
  DebugMode debugMode() const { return dm_; }

  static DebugMode defaultDebugMode() { return DebugMode::Off; }

  static RotationAlgo defaultRotationAlgo() { return RotationAlgo::RIPPLE; }

  static double defaultRotationLimitSeconds() { return 1e9; }

  static int64_t defaultRotationLimitCount() { return 1000000000; }

  static RotationTermination defaultRotationTermination();

  static KahnTieBreaker defaultKahnTieBreaker();

  static uint32_t defaultSeed() { return 1; }

  static TransitiveClosureOptimizations defaultTCOs();

  std::tuple<KahnTieBreaker,
             TransitiveClosureOptimizations,
             RotationTermination,
             RotationAlgo,
             uint32_t,
             DebugMode>
  getTuple() const {
    return {ktb_, tcos_, rt_, ra_, seed_, dm_};
  }

  bool operator==(const Settings &rhs) const {
    return getTuple() == rhs.getTuple();
  }
  bool operator<(const Settings &rhs) const {
    return getTuple() < rhs.getTuple();
  }

private:
  KahnTieBreaker ktb_;
  TransitiveClosureOptimizations tcos_;
  RotationTermination rt_;
  RotationAlgo ra_;
  uint32_t seed_;
  DebugMode dm_;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
