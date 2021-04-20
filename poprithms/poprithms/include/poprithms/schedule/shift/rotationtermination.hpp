// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_ROTATIONTERMINATION
#define POPRITHMS_SCHEDULE_SHIFT_ROTATIONTERMINATION

#include <limits>
#include <numeric>
#include <tuple>

namespace poprithms {
namespace schedule {
namespace shift {

class RotationTermination {
public:
  RotationTermination(double t_, int64_t r_)
      : tSeconds_(t_), maxRotations_(r_) {}

  static RotationTermination preStart() { return {-1.0, -1}; }
  static RotationTermination nHours(int n) {
    return {60. * 60. * n, std::numeric_limits<int64_t>::max()};
  }

  void setMaxSeconds(double s) { tSeconds_ = s; }
  void setMaxRotations(int64_t r) { maxRotations_ = r; }

  double maxSeconds() const { return tSeconds_; }
  int64_t maxRotations() const { return maxRotations_; }

  std::tuple<double, int64_t> getTuple() const {
    return {tSeconds_, maxRotations_};
  }

  bool operator==(const RotationTermination &rhs) const {
    return getTuple() == rhs.getTuple();
  }
  bool operator<(const RotationTermination &rhs) const {
    return getTuple() < rhs.getTuple();
  }

private:
  double tSeconds_;
  int64_t maxRotations_;
};

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
