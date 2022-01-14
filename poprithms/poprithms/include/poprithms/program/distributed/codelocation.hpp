// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_DISTRIBUTED_CODELOCATION_HPP
#define POPRITHMS_PROGRAM_DISTRIBUTED_CODELOCATION_HPP

#include <ostream>
#include <vector>

namespace poprithms {
namespace program {
namespace distributed {

enum class CodeLocation { None, Ipu, Host };
std::ostream &operator<<(std::ostream &, CodeLocation);

} // namespace distributed
} // namespace program
} // namespace poprithms

#endif
