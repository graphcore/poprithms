// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_NUMPYFORMATTER_HPP
#define POPRITHMS_COMPUTE_HOST_NUMPYFORMATTER_HPP

#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace compute {
namespace host {

/**
 * A class to help format Tensor data strings, to resemble numpy behaviour.
 */
class NumpyFormatter {
public:
  /**
   * A string formatting method similar to numpy.ndarray's style:
   *
   *                    [[ 1  11 3 4 ]
   *                     [ 10 2  7 5 ]].
   *
   * \param rowMajorElements The elements of the Tensor to display, in
   *                         row-major order,
   *
   * \param toAppendTo The stream to append the Tensor representation to.
   *
   * \param shape The Shape of the Tensor. The number of elements in \p
   *              rowMajorElements must equal the number of elements in
   *              \p shape.
   *
   * \param abbrevThreshold If the number of elements exceeds this value, an
   *                        incomplete representation of the Tensor is
   *                        produced.
   * */
  static void append(const std::vector<std::string> &rowMajorElements,
                     std::ostream &toAppendTo,
                     const ndarray::Shape &shape,
                     uint64_t abbreviationThreshold = 100);
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
