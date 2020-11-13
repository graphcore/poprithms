// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef POPRITHMS_MEMORY_INPLACE_ALIAS_TYPE_HPP
#define POPRITHMS_MEMORY_INPLACE_ALIAS_TYPE_HPP

#include <cstdint>
#include <sstream>

namespace poprithms {
namespace memory {
namespace inplace {

/**
 * High-level description of how inputs and outputs are aliased.
 * */
class AliasType {
public:
  /**
   * A general purpose AliasType for operators which do not have
   * inplace/outplace variants.
   * */
  static AliasType none() { return AliasType(TypeEnum::None); }

  /**
   * AliasType where there is no aliasing between inputs and outputs.
   * */
  static AliasType outplace() { return AliasType(TypeEnum::Outplace); }

  /** AliasType where all elements of all inputs are aliased to at least one
   * element of an output. */
  static AliasType allInplace() { return AliasType(TypeEnum::AllInplace); }

  /** AliasType specific to binary operators, where the output aliases the
   * first input argument */
  static AliasType binary0() { return AliasType(TypeEnum::Binary0); }

  /** AliasType specific to binary operators, where the output aliases the
   * second input argument */
  static AliasType binary1() { return AliasType(TypeEnum::Binary1); }

  static AliasType binary(uint64_t index) {
    assertZeroOrOne(index);
    return index == 0 ? binary0() : binary1();
  }

  bool isOutplace() const { return type_ == TypeEnum::Outplace; }

  bool isAllInplace() const { return type_ == TypeEnum::AllInplace; }

  bool isBinary0() const { return type_ == TypeEnum::Binary0; }

  bool isBinary1() const { return type_ == TypeEnum::Binary1; }

  bool isNone() const { return type_ == TypeEnum::None; }

  bool operator==(const AliasType &rhs) const { return type_ == rhs.type_; }

  bool operator!=(const AliasType &rhs) const { return !operator==(rhs); }

  void append(std::ostream &ost) const;

private:
  enum class TypeEnum { Outplace, AllInplace, Binary0, Binary1, None };
  AliasType(TypeEnum i) : type_(i) {}
  static void assertZeroOrOne(uint64_t);
  TypeEnum type_;
};

std::ostream &operator<<(std::ostream &, AliasType);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
