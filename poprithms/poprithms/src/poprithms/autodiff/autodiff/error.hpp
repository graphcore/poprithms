// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_AUTODIFF_AUTODIFF_ERROR_HPP
#define POPRITHMS_AUTODIFF_AUTODIFF_ERROR_HPP

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace autodiff {

poprithms::error::error error(const std::string &what);

}
} // namespace poprithms

#endif
