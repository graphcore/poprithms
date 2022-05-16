// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_CARRIEDTENSORID_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_CARRIEDTENSORID_HPP

#include <algorithm>
#include <vector>

#include <poprithms/common/multiout/tensorid.hpp>

namespace poprithms {
namespace program {
namespace callstack {

using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;

/**
 * Tuple of 3 tensors describing a loop carried dependency.
 *
 * 1) copy from the caller into the callee sub-graph:
 * copy(sourceInCaller, destinationInCallee)
 * for iter in range(nIterations):
 *    run(callee)
 *    2) between iterations, copy the loop carried dependency:
 *    copy(sourceInCallee, destinationInCallee)
 *
 *
 *       sourceInCaller
 *             |
 *   . . . . . | . . . . . . .
 *   .         v             .
 *   . destinationInCallee   .
 *   .                       . iteration 0
 *   .    sourceInCallee     .
 *   . . . . . | . . . . . . .
 *             |
 *   . . . . . | . . . . . . .
 *   .         v             .
 *   . destinationInCallee   .
 *   .                       . iteration 1
 *   .    sourceInCallee     .
 *   . . . . . | . . . . . . .
 *             |
 *            etc.
 * */
class CarriedTensorId {
public:
  CarriedTensorId() = default;
  CarriedTensorId(const TensorId &sourceInCaller,
                  const TensorId &destinationInCallee,
                  const TensorId &sourceInCallee)
      : sourceInCaller_(sourceInCaller),
        destinationInCallee_(destinationInCallee),
        sourceInCallee_(sourceInCallee) {}

  TensorId sourceInCaller() const { return sourceInCaller_; }
  TensorId destinationInCallee() const { return destinationInCallee_; }
  TensorId sourceInCallee() const { return sourceInCallee_; }

  void append(std::ostream &) const;

private:
  TensorId sourceInCaller_;
  TensorId destinationInCallee_;
  TensorId sourceInCallee_;
};

/**
 * Multiple CarriedTensorIds.
 * */
class CarriedTensorIds {
public:
  CarriedTensorIds() = default;
  CarriedTensorIds(const std::vector<CarriedTensorId> &ts)
      : carriedTensorIds_(ts) {}
  CarriedTensorIds(std::vector<CarriedTensorId> &&ts)
      : carriedTensorIds_(std::move(ts)) {}

  static CarriedTensorIds zip(const TensorIds &sourcesInCaller,
                              const TensorIds &destinationsInCallee,
                              const TensorIds &sourcesInCallee);

  const std::vector<CarriedTensorId> &carriedTensorIds() const {
    return carriedTensorIds_;
  }

  const CarriedTensorId &carriedTensorId(uint64_t i) const {
    return carriedTensorIds_.at(i);
  }

  uint64_t nTensors() const { return carriedTensorIds_.size(); }

  TensorIds sourcesInCallee() const;

private:
  std::vector<CarriedTensorId> carriedTensorIds_;
};

std::ostream &operator<<(std::ostream &, const CarriedTensorId &);
std::ostream &operator<<(std::ostream &, const CarriedTensorIds &);
} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
