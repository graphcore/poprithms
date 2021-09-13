// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <schedule/shift/updatefromfirstfinal.hpp>

#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/logging.hpp>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>

namespace poprithms {
namespace schedule {
namespace shift {

namespace {

void updateFromFirst(AllocWeight &lwr,
                     AllocWeight &upp,
                     const AllocWeight &w,
                     const transitiveclosure::IsFirst isFirst) {
  switch (isFirst) {
  // If an Op is definitely not the first consumer of an allocation, the
  // allocation definitely does not increase liveness
  case (transitiveclosure::IsFirst::No): {
    break;
  }
  case (transitiveclosure::IsFirst::Maybe): {
    // If an Op might be the first consumer of an allocation, the allocation
    // might increase liveness. The upper-bound on liveness is therefore
    // increased
    upp += w;
    break;
  }
  case (transitiveclosure::IsFirst::Yes): {
    lwr += w;
    upp += w;
    break;
  }
  }
}

void updateFromFinal(AllocWeight &lwr,
                     AllocWeight &upp,
                     const AllocWeight &w,
                     const transitiveclosure::IsFinal isFinal) {
  switch (isFinal) {
  case (transitiveclosure::IsFinal::No): {
    break;
  }
  case (transitiveclosure::IsFinal::Maybe): {
    lwr -= w;
    break;
  }
  case (transitiveclosure::IsFinal::Yes): {
    lwr -= w;
    upp -= w;
    break;
  }
  }
}

} // namespace

void updateFromFirstFinal(AllocWeight &lwr,
                          AllocWeight &upp,
                          const AllocWeight &w,
                          const std::tuple<transitiveclosure::IsFirst,
                                           transitiveclosure::IsFinal> ff) {
  updateFromFirst(lwr, upp, w, std::get<0>(ff));
  updateFromFinal(lwr, upp, w, std::get<1>(ff));
}

} // namespace shift
} // namespace schedule
} // namespace poprithms
