// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_ALLOW_MULTI_GATE_ALIAS_HPP
#define POPRITHMS_MEMORY_INPLACE_ALLOW_MULTI_GATE_ALIAS_HPP

#include <ostream>

namespace poprithms {
namespace memory {
namespace inplace {

/**
 * \bf Definition
 * --------------
 *
 * Consider an alias gate with 3 inputs, open at index 1:
 *
 *  in0 -----------+
 *                 |
 *  in1 - open  ---+----->  out.
 *                 |
 *  in2 -----------+
 *
 * #out is an exact alias of #in1 (by definition of an alias gate). 
 *
 * If #in1 is aliased to #in0, then #out is also aliased to to #in0.
 * Similarly, if #in1 is aliased to #in2, then #out is aliased to #in2.
 *
 * This enum class defines if this 'multi' aliasing is allowed. Specifically,
 * is the output of an alias gate ever allowed to alias an input other than
 * the one at the open index? Equivalently, can the input at the open index be
 * aliased to any other inputs?
 *
 * During inplacing, if a proposed opening causes any alias gate's output to
 * alias more than one input, the proposal is rejected, if
 * AllowMultiGateAlias::No.
 *
 * \bf Use case
 * ------------
 *
 * The motivation for this enum is to avoid race conditions. One use of alias
 * gates is to represent binary elementwise operations, such as 'mul'.
 * Consider the case:
 *
 * > in  = 1-d tensor with values (2,3).
 * > out = in.mul(in.reverse())
 *       = 1-d tensor with values (6,6).
 *
 * If this were replaced with an inplace multiplication, then a race
 * condition would arise. If the elements are processed in[0] then in[1], then
 * the output is
 *
 * > out = in.mul_(in.reverse()),
 *       = 1-d tensor with values (6, 18),
 *
 * and if the elements are processed in[1] then in[0], the output is
 *
 * > out = in.mul_(in.reverse()),
 *       = 1-d tensor with values (12, 6).
 *
 * And so the order of processing matters, a situation which we refer to as a
 * 'race condition'.
 *
 * If we disallow all cases of a.mul_(b) where 'a' and 'b' are aliased, we
 * will avoid this race condition. Note that this is conservative, as there is
 * no race condition in a.mul_(a). Specifically, "race condition" implies
 * "inputs are aliased", but "inputs are aliased" does not imply "race
 * condition". Detecting exactly when race conditions arise is beyond the
 * scope of this project, as it requires information about element ordering.
 *
 *  */

enum class AllowMultiGateAlias {
  No = 0, ///< Do not allow alias gate outputs to alias multiple inputs.
  Yes     ///< Allow alias gate outputs to alias multiple inputs.
};

std::ostream &operator<<(std::ostream &, AllowMultiGateAlias);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
