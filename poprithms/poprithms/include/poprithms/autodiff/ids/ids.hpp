// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_IDS_IDS_HPP
#define POPRITHMS_AUTODIFF_IDS_IDS_HPP

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/optraversal.hpp>
#include <poprithms/common/multiout/tensorid.hpp>

namespace poprithms {
namespace autodiff {

/**
 * We borrow these small classes from the common::multiout project. Note that
 * we don't use the multiout::Graph class in this autodiff project, as we want
 * to keep it as independent as possible from a graph structure.
 * */

// three integral types.
using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OutIndex;

// (InIndex, OpId)
using poprithms::common::multiout::ConsumptionId;

// (OpId, OutIndex)
using poprithms::common::multiout::TensorId;

// either (OpId, OutIndex) or "not set"
using poprithms::common::multiout::OptionalTensorId;

// (InIndex, OpId, OutIndex).
using poprithms::common::multiout::OpTraversal;

// vectors of all of the above
using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::OpTraversals;
using poprithms::common::multiout::TensorIds;

} // namespace autodiff
} // namespace poprithms

#endif
