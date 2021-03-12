// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_CHAIN_TYPE_HPP
#define POPRITHMS_MEMORY_CHAIN_TYPE_HPP

#include <ostream>

namespace poprithms {
namespace memory {
namespace chain {
/** Unlike most view-changing Graph projects in poprithms, the Chain project
 * does not use polymorphism for the different Op types. This is to facilitate
 * the template method "get" in the Graph class, used by the public template
 * method "apply".
 *
 * Instead of using polymorphism, each Op has an enum to describe how it
 * changes the view of a Tensor.
 * */
enum class Type {

  DimShuffle =
      0, ///< A DimShuffle Op is a generalization of a 2-D
         ///< transpose to higher dimensions. It has a Permutation attribute
         ///< which defines how the dimensions are shuffled.

  Expand, ///< An Expand Op broadcasts a Tensor in certain singleton
          ///< dimensions. It has a Shape attribute, which defines the output
          ///< Shape, which implicitly defines which dimensions are broadcast.
          ///< Any Shape which can be added to the input using numpy
          ///< broadcasting rules is a valid Shape attribute.

  Reduce, ///< A Reduce Op is the inverse of Expand, which performs a
          ///< reduction along certain dimensions. It has a Shape attribute,
          ///< which implicitly defines the dimensions which are reduced. We
          ///< assume here that all Reduces are by summation. TODO(T35649)
          ///< rethink this.

  Reshape, ///< A Reshape Op reshapes a Tensor. It has a Shape attribute,
           ///< which is the Shape of the output. The one constraint on the
           ///< Shape is that the number of elements is unchanged from the
           ///< input.

  Reverse, ///< A Reverse Op reverses a Tensor along certain dimensions. It
           ///< has a Dimensions attribute, which defines the dimensions of
           ///< the input Tensor to reverse.

  SettSample, ///< A SettSample Op is a generalization of slice and
              ///< subSample. It has a Region attribute, which defines the
              ///< elements to retain in the output.

  SettFillInto ///< A SettFillInto Op is the inverse of SettSample, and it
               ///< scatters the values of its input into a new Tensor. It has
               ///< a Region attribute, which defines the locations in the
               ///< output to which the input is scattered.
};

std::ostream &operator<<(std::ostream &, Type);
std::string getTypeString(Type);

} // namespace chain
} // namespace memory
} // namespace poprithms

#endif
