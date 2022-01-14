// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <poprithms/program/distributed/codelocation.hpp>

namespace poprithms {
namespace program {
namespace distributed {

std::ostream &operator<<(std::ostream &ost, CodeLocation cl) {
  switch (cl) {
  case CodeLocation::Host: {
    ost << "Host";
    return ost;
  }
  case CodeLocation::Ipu: {
    ost << "Ipu";
    return ost;
  }
  case CodeLocation::None: {
    ost << "None";
    return ost;
  }
  }

  throw error("unrecognised CodeLocation in operator<<");
}

} // namespace distributed
} // namespace program
} // namespace poprithms
