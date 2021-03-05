// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_SUMLIKE_HPP
#define POPRITHMS_MEMORY_UNWIND_SUMLIKE_HPP

#include <sstream>

#include <poprithms/common/multiout/tensorid.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using common::multiout::OpId;
using common::multiout::TensorId;

/**
 *  Example: An addition of 2 Tensors using numpy broadcating:
 *
 *  aaa   +  b     =  ccc
 *  aaa      b        ccc
 *
 *  where the Shapes are:
 *   a       b        c
 *  (2,3) + (2,1) -> (2,3).
 *
 *  The layout of `b` can be derived from `a`, along the lines of Poplibs's
 *  createBias. This is handled as follows:
 *
 *  A barrier op takes `a` as input, and outputs `d` of shape (2,1). `d`'s
 *  layout is then set by the user (poplibs). An attraction between `d` and
 * `b` is inserted to encourage `b` to have the same layout as `d`.
 *
 *  The triplet (a, b, d) is captured in the following class where the
 *  correspondence between member methods with the above example is:
 *
 *  sumLikeInput : a
 *  reduced      : d
 *  target       : b
 *  */

class SumLikeMapping {
public:
  SumLikeMapping(const TensorId &sumLikeInput,
                 OpId barrier,
                 const TensorId target)
      : sumLikeInput_(sumLikeInput), barrierOpId_(barrier), target_(target) {}
  TensorId sumLikeInput() const { return sumLikeInput_; }
  OpId barrier() const { return barrierOpId_; }
  TensorId reduced() const { return {barrier(), 0}; }
  TensorId target() const { return target_; }
  void append(std::ostream &) const;

private:
  TensorId sumLikeInput_;
  OpId barrierOpId_;
  TensorId target_;
};

std::ostream &operator<<(std::ostream &, const SumLikeMapping &);

using SumLikeMappings = std::vector<SumLikeMapping>;

/**
 * The output of the sumLike operation, consisting of (1) the output Tensor
 * and (2) all of the potential layout mappings between inputs of different
 * sizes.
 * */
class SumLikeOut {
public:
  SumLikeOut(const TensorId &out, const SumLikeMappings &mappings)
      : out_(out), mappings_(mappings) {}

  /** The output of the sumLike operation. */
  TensorId out() const { return out_; }

  /** All of the possible ways one input can determine the layout of another
   * input of a different size.  */
  const SumLikeMappings &mappings() const { return mappings_; }

  const SumLikeMapping &mapping(uint64_t i) const { return mappings_[i]; }
  const TensorId reduced(uint64_t i) const { return mapping(i).reduced(); }
  const TensorId target(uint64_t i) const { return mapping(i).target(); }
  const OpId barrier(uint64_t i) const { return mapping(i).barrier(); }

  void append(std::ostream &) const;

private:
  TensorId out_;
  SumLikeMappings mappings_;
};

std::ostream &operator<<(std::ostream &, const SumLikeOut &);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
