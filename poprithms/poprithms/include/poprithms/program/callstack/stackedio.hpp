// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_STACKEDIO_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_STACKEDIO_HPP

namespace poprithms {
namespace program {
namespace callstack {

enum class IsStackedCopy { No = 0, Yes };
enum class StackedCopyOrder { Up = 0, Down };

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
