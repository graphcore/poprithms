// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_TENSORUSINGS_HPP
#define POPRITHMS_COMPUTE_HOST_TENSORUSINGS_HPP
#include <memory>
#include <sstream>

#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace compute {
namespace host {

using Shape       = poprithms::ndarray::Shape;
using DType       = poprithms::ndarray::DType;
using Permutation = poprithms::util::Permutation;
using Lower       = Shape::Lower;
using Upper       = Shape::Upper;
using Shapes      = std::vector<Shape>;

class Tensor;
using Tensors = std::vector<Tensor>;

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
