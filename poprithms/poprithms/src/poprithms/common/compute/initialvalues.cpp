// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <sstream>

#include <poprithms/common/compute/initialvalues.hpp>

namespace poprithms {
namespace common {
namespace compute {

void InitialValues::setValue(OutIndex o,
                             uint64_t replica,
                             const HostTensor &initVal) {

  auto &replicaValues = chts.at(o.get());

  auto found = replicaValues.find(replica);
  if (found != replicaValues.cend()) {

    poprithms::compute::host::Tensor existingVal = found->second.tensor();

    std::ostringstream oss;
    oss << "Setting initial value at output index #" << o << " of replica #"
        << replica << ", but there is already a value set. "
        << "Conservatively throwing here, "
        << "as this is potentially a user error "
        << "(please report if not and this error will be removed). "
        << "Value currently set is " << existingVal
        << " and proposed new value is " << initVal << ".";

    throw error(oss.str());
  }

  replicaValues.insert({replica, initVal});
}

std::map<uint64_t, HostTensor>
InitialValues::getInitialValues(OutIndex o) const {

  if (o.get() >= nOutTensors()) {
    std::ostringstream oss;
    oss << "Invalid output index (" << o << ") with only " << nOutTensors()
        << " output tensors.";
    throw error(oss.str());
  }

  std::map<uint64_t, HostTensor> ts;
  for (const auto &[r, t] : chts.at(o.get())) {
    ts.insert({r, t.tensor()});
  }
  return ts;
}

} // namespace compute
} // namespace common
} // namespace poprithms
