// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_UTIL_STRINGUTIL_HPP
#define POPRITHMS_UTIL_STRINGUTIL_HPP

#include <string>

namespace poprithms {
namespace util {

std::string lowercase(const std::string &);

/**
 * A logging utility for string aligment. This function returns a string of
 * ' ' of length max(1, target - ts.size() + 1)
 * */
std::string spaceString(uint64_t target, const std::string &ts);

} // namespace util
} // namespace poprithms

#endif
