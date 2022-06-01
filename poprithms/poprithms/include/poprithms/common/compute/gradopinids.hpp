// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_GRADOPINIDS_HPP
#define POPRITHMS_COMMON_COMPUTE_GRADOPINIDS_HPP

#include <poprithms/autodiff/automatic/gradopin.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>

namespace poprithms {
namespace common {
namespace compute {

using poprithms::common::multiout::OptionalTensorId;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;

using GradOpInIds =
    poprithms::autodiff::automatic::OpIn<TensorId, OptionalTensorId>;

} // namespace compute
} // namespace common
} // namespace poprithms
#endif
