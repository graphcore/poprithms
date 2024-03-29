// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <cctype>
#include <sstream>
#include <string>

#include <ndarray/error.hpp>

#include <poprithms/ndarray/dtype.hpp>

namespace poprithms {
namespace ndarray {

namespace {

constexpr uint64_t asUnsigned64(DType t) { return static_cast<uint64_t>(t); }

enum class NonNegative { No = 0, Yes };

struct NumericTypeInfo {
public:
  DType type;
  int nbytes;
  bool isFixedPoint;
  NonNegative isNonNegative;
  std::string pcase;
  std::string lcase;

  bool nonNegative() const { return isNonNegative == NonNegative::Yes; }

  NumericTypeInfo() = default;

  NumericTypeInfo(DType type__,
                  int nb__,
                  bool ifp__,
                  NonNegative inn__,
                  std::string pcase__)
      : type(type__), nbytes(nb__), isFixedPoint(ifp__), isNonNegative(inn__),
        pcase(pcase__) {

    lcase = pcase;
    std::transform(lcase.begin(), lcase.end(), lcase.begin(), [](char c) {
      return std::tolower(c);
    });
  }
};

constexpr auto NTypes = asUnsigned64(DType::N);

// Used to internally check that all types are set.
constexpr const char *unsetCase = "___unset___case___";

std::array<NumericTypeInfo, NTypes> initInfoArray() {

  NumericTypeInfo undefined(
      DType::N, -100, false, NonNegative::No, unsetCase);
  std::array<NumericTypeInfo, NTypes> infos;
  infos.fill(undefined);

  auto registerInfo = [&infos](DType t,
                               int numberOfBytes,
                               bool isFixedPoint,
                               NonNegative isNonNegative,
                               std::string upperCaseName) {
    infos[asUnsigned64(t)] = {
        t, numberOfBytes, isFixedPoint, isNonNegative, upperCaseName};
  };

  registerInfo(DType::Float16, 2, false, NonNegative::No, "Float16");
  registerInfo(DType::Float32, 4, false, NonNegative::No, "Float32");
  registerInfo(DType::Float64, 8, false, NonNegative::No, "Float64");

  registerInfo(DType::Int8, 1, true, NonNegative::No, "Int8");
  registerInfo(DType::Int16, 2, true, NonNegative::No, "Int16");
  registerInfo(DType::Int32, 4, true, NonNegative::No, "Int32");
  registerInfo(DType::Int64, 8, true, NonNegative::No, "Int64");

  registerInfo(
      DType::Boolean, sizeof(bool), true, NonNegative::Yes, "Boolean");
  registerInfo(DType::Unsigned8, 1, true, NonNegative::Yes, "Unsigned8");
  registerInfo(DType::Unsigned16, 2, true, NonNegative::Yes, "Unsigned16");
  registerInfo(DType::Unsigned32, 4, true, NonNegative::Yes, "Unsigned32");
  registerInfo(DType::Unsigned64, 8, true, NonNegative::Yes, "Unsigned64");

  // Check that all the DTypes have been registered.
  for (auto &v : infos) {
    if (v.pcase == unsetCase) {
      std::ostringstream oss;
      oss << "Failure in initInfoArray. "
          << "Not all DTypes have valid settings. "
          << "Has a new DType been added, without it being described? "
          << "Bailing.";
      throw error(oss.str());
    }
  }

  return infos;
}

const std::array<NumericTypeInfo, NTypes> &getInfoArray() {
  const static auto infos = initInfoArray();
  return infos;
}

} // namespace

std::ostream &operator<<(std::ostream &os, DType t) {
  os << lcase(t);
  return os;
}

int nbytes(DType dtype) {
  return getInfoArray().at(asUnsigned64(dtype)).nbytes;
}

uint64_t nbytes_u64(DType dtype) {
  return static_cast<uint64_t>(nbytes(dtype));
}

const std::string &lcase(DType dtype) {
  return getInfoArray().at(asUnsigned64(dtype)).lcase;
}

const std::string &pcase(DType dtype) {
  return getInfoArray().at(asUnsigned64(dtype)).pcase;
}

bool isFixedPoint(DType dtype) {
  return getInfoArray().at(asUnsigned64(dtype)).isFixedPoint;
}

bool isNonNegative(DType dtype) {
  return getInfoArray().at(asUnsigned64(dtype)).nonNegative();
}

template <typename T> void verify(DType t) {
  if (t != get<T>()) {
    std::ostringstream ss;
    ss << "Failure in verify<T>(" << lcase(t) << "). Expected "
       << lcase(get<T>()) << '.';
    throw error(ss.str());
  }
}

bool isUnsignedFixedPoint(DType t) {
  return isFixedPoint(t) && isNonNegative(t);
}

template <> DType get<double>() { return DType::Float64; }
template void verify<double>(DType);

template <> DType get<float>() { return DType::Float32; }
template void verify<float>(DType);

template <> DType get<uint8_t>() { return DType::Unsigned8; }
template void verify<uint8_t>(DType);

template <> DType get<uint16_t>() { return DType::Unsigned16; }
template void verify<uint16_t>(DType);

template <> DType get<uint32_t>() { return DType::Unsigned32; }
template void verify<uint32_t>(DType);

template <> DType get<uint64_t>() { return DType::Unsigned64; }
template void verify<uint64_t>(DType);

template <> DType get<int8_t>() { return DType::Int8; }
template void verify<int8_t>(DType);

template <> DType get<int16_t>() { return DType::Int16; }
template void verify<int16_t>(DType);

template <> DType get<int32_t>() { return DType::Int32; }
template void verify<int32_t>(DType);

template <> DType get<int64_t>() { return DType::Int64; }
template void verify<int64_t>(DType);

template <> DType get<bool>() { return DType::Boolean; }
template void verify<bool>(DType);

} // namespace ndarray
} // namespace poprithms
