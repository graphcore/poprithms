// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_ALIAS_TYPE_HPP
#define POPRITHMS_MEMORY_INPLACE_ALIAS_TYPE_HPP

#include <cstdint>
#include <sstream>

namespace poprithms {
namespace memory {
namespace inplace {

class AliasType {
public:
  /**
   * AliasType where there is no aliasing between inputs and outputs
   * */
  static AliasType outplace() { return AliasType(TypeEnum::Out); }

  /** AliasType where all elements of all inputs are aliased to at least one
   * element of an output */
  static AliasType all() { return AliasType(TypeEnum::All); }

  /** AliasType specific to binary operators, where the output aliases the
   * first input argument */
  static AliasType binary0() { return AliasType(TypeEnum::Binary0); }

  /** AliasType specific to binary operators, where the output aliases the
   * second input argument */
  static AliasType binary1() { return AliasType(TypeEnum::Binary1); }

  static AliasType binary(uint64_t index) {
    return index == 0 ? binary0() : binary1();
  }

  bool isOutplace() const { return type_ == TypeEnum::Out; }

  bool isInplace() const { return !isOutplace(); }

  bool operator==(const AliasType &rhs) const { return type_ == rhs.type_; }

  bool operator!=(const AliasType &rhs) const { return !operator==(rhs); }

  void append(std::ostream &ost) const;

private:
  enum class TypeEnum { Out, All, Binary0, Binary1 };
  AliasType(TypeEnum i) : type_(i) {}
  TypeEnum type_;
};

std::ostream &operator<<(std::ostream &, AliasType);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
