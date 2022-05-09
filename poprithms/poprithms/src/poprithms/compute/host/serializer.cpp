// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/container/vector.hpp>
#include <boost/operators.hpp>
#include <boost/serialization/boost_array.hpp>
#include <boost/serialization/boost_unordered_map.hpp>
#include <boost/serialization/boost_unordered_set.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/vector.hpp>

#include <compute/host/include/allocdata.hpp>
#include <compute/host/include/basedata.hpp>
#include <compute/host/include/baseoperators.hpp>
#include <compute/host/include/externdecl.hpp>
#include <compute/host/include/ieeehalf.hpp>
#include <compute/host/include/origindata.hpp>
#include <compute/host/include/pointerdata.hpp>
#include <compute/host/include/typeddata.hpp>
#include <compute/host/include/viewdata.hpp>
#include <compute/host/serializer.hpp>
#include <ndarray/serializer.hpp>

#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/error/error.hpp>

namespace poprithms {
namespace ndarray {

template <typename Archive>
// Serialize a Shape object
void BoostSerializer::serialize(Archive &a,
                                ndarray::Shape &shape,
                                uint32_t version) {
  (void)version;
  // Access to private member 'shp' granted through friendship.
  a &shape.shp;
}

template void BoostSerializer::serialize(boost::archive::text_oarchive &,
                                         ndarray::Shape &,
                                         uint32_t);

template void BoostSerializer::serialize(boost::archive::text_iarchive &,
                                         ndarray::Shape &,
                                         uint32_t);
} // namespace ndarray
} // namespace poprithms

namespace poprithms {
namespace compute {
namespace host {

class Serializer {

public:
  template <typename Archive>
  static void
  serialize(Archive &, compute::host::BaseData &, uint32_t /* version */) {
    // No attributes in this virtual base class (BaseData), no serialization
    // needed.
  }

  template <typename Archive, typename T>
  static void serialize(Archive &,
                        compute::host::TypedData<T> &,
                        uint32_t /* version */) {
    // No attributes in this virtual base class.
  }

  template <typename Archive, typename T>
  static void serialize(Archive &,
                        compute::host::PointerData<T> &ht,
                        uint32_t /* version */) {
    std::ostringstream oss;
    oss << "Cannot serialize poprithms::compute::host::Tensor with "
        << "underlying data of type 'PointerData'. "
        << "The pointer to the data is not owned by the Tensor. "
        << "It would be possible to copy the data to a new buffer and take "
        << "ownership, but the alias semantics of the serialized tensors "
        << "would then change so current implementation does not do this. "
        << "Recommendation: do not serialize Tensors unless they own their "
        << "data. Please open a ticket if this is not possible to change. "
        << "\nPointerData is: " << ht;
    throw error(oss.str());
  }

  template <typename Archive, typename T>
  static void serialize(Archive &, compute::host::OriginData<T> &, uint32_t) {
    // No attributes in this virtual base class.
  }

  template <typename Archive, typename T>
  static void save(Archive &a,
                   const compute::host::ViewData<T> &bd,
                   uint32_t /* version */) {

    // A ViewData<T> object has 4 fields, 3 which are passed into the
    // ViewData<T> constructor, and a fourth field (rowMajorOriginDataPtrs)
    // which is a vector<T *> derived from the first. It is impossible to
    // correctly serialize these pointers ahead of time. For this reason, we
    // have split the serialize method into a 'save' and 'load' as
    // described and recommended in the boost docs.
    //
    // We only save the first 3 fields, the vector<T *> will be computed in
    // 'load' based on the loaded values.

    a << bd.rowMajorOriginDatas;
    a << bd.rowMajorOriginDataIndices;
    a << bd.rowMajorOriginDataOffsets;
  }

  template <typename Archive, typename T>
  static void
  load(Archive &a, compute::host::ViewData<T> &bd, uint32_t version) {
    // Load the 3 serialized fields and set the fourth manually.
    (void)version;
    a >> bd.rowMajorOriginDatas;
    a >> bd.rowMajorOriginDataIndices;
    a >> bd.rowMajorOriginDataOffsets;
    bd.setRowMajorOriginDataPtrs();
  }

  template <typename Archive, typename T>
  static void serialize(Archive &a,
                        compute::host::AllocData<T> &bd,
                        uint32_t /* version */) {

    // AllocData has 1 attribute, a unique pointer to the underlying data.
    a &bd.up;
  }

  template <typename Archive>
  static void
  serialize(Archive &a, compute::host::Tensor &t, uint32_t version) {
    (void)version;
    a &t.shape_;
    a &t.dtype_;
    a &t.tData_;
  }
};

} // namespace host
} // namespace compute
} // namespace poprithms

namespace boost {
namespace serialization {

template <typename Archive>
void serialize(Archive &a,
               compute::host::BaseData &bd,
               const uint32_t version) {
  (void)version;
  poprithms::compute::host::Serializer::serialize<Archive>(a, bd, version);
}

template <typename Archive, typename T>
void save(Archive &a,
          const compute::host::ViewData<T> &od,
          const uint32_t version) {
  a &boost::serialization::base_object<compute::host::TypedData<T>>(od);
  poprithms::compute::host::Serializer::save<Archive>(a, od, version);
}

template <typename Archive, typename T>
void load(Archive &a,
          compute::host::ViewData<T> &od,
          const uint32_t version) {
  a &boost::serialization::base_object<compute::host::TypedData<T>>(od);
  poprithms::compute::host::Serializer::load<Archive>(a, od, version);
}

template <typename Archive, typename T>
void serialize(Archive &a,
               compute::host::ViewData<T> &od,
               const uint32_t version) {
  // Following the boost guidelines for serializing a class which requires
  // different handling of load and save.
  boost::serialization::split_free(a, od, version);
}

template <typename Archive, typename T>
void serialize(Archive &a,
               compute::host::TypedData<T> &od,
               const uint32_t version) {
  (void)version;
  a &boost::serialization::base_object<compute::host::BaseData>(od);
  poprithms::compute::host::Serializer::serialize<Archive>(a, od, version);
}

template <typename Archive, typename T>
void serialize(Archive &a,
               compute::host::OriginData<T> &od,
               const uint32_t version) {
  (void)version;
  a &boost::serialization::base_object<compute::host::TypedData<T>>(od);
  poprithms::compute::host::Serializer::serialize<Archive>(a, od, version);
}

template <typename Archive, typename T>
void serialize(Archive &a,
               compute::host::PointerData<T> &od,
               const uint32_t version) {
  (void)version;
  a &boost::serialization::base_object<compute::host::OriginData<T>>(od);
  poprithms::compute::host::Serializer::serialize<Archive>(a, od, version);
}

template <typename Archive, typename T>
void serialize(Archive &a,
               compute::host::AllocData<T> &od,
               const uint32_t version) {
  (void)version;
  a &boost::serialization::base_object<compute::host::OriginData<T>>(od);
  poprithms::compute::host::Serializer::serialize<Archive>(a, od, version);
}

// Required as no default constructor for AllocData
template <class Archive, class T>
inline void
load_construct_data(Archive &ar, compute::host::AllocData<T> *t, uint32_t v) {
  (void)ar;
  (void)v;
  ::new (t) compute::host::AllocData<T>(std::vector<T>{});
}

// Required as no default constructor for ViewData
template <class Archive, class T>
inline void
load_construct_data(Archive &ar, compute::host::ViewData<T> *t, uint32_t v) {
  (void)ar;
  (void)v;
  ::new (t) compute::host::ViewData<T>({}, {}, {});
}

// Required as no default constructor for PointerData
template <class Archive, class T>
inline void load_construct_data(Archive &ar,
                                compute::host::PointerData<T> *t,
                                uint32_t v) {
  (void)ar;
  (void)v;
  ::new (t) compute::host::PointerData<T>({}, {});
}

// boost::container::vectors do not have native serialization support, so we
// convert to a std::vector and serialize. As std::vector<bool> is a compact
// form this is a very efficient way to serialize.
template <typename Archive>
void save(Archive &a,
          const boost::container::vector<bool> &bd,
          uint32_t /* version */) {
  std::vector<bool> v(bd.cbegin(), bd.cend());
  a << v;
}

template <typename Archive>
void load(Archive &a,
          boost::container::vector<bool> &bd,
          uint32_t /* version */) {
  std::vector<bool> v(bd.cbegin(), bd.cend());
  a >> v;
  bd = {v.cbegin(), v.cend()};
}

template <typename Archive>
void serialize(Archive &a,
               boost::container::vector<bool> &v,
               const uint32_t version) {
  boost::serialization::split_free(a, v, version);
}

// Support for float16s. See the test, testTwoFloat16
template <typename Archive>
void save(Archive &a,
          const copied_from_poplar::IeeeHalf &bd,
          uint32_t /* version */) {
  a << bd.bit16();
}

template <typename Archive>
void load(Archive &a,
          copied_from_poplar::IeeeHalf &bd,
          uint32_t /* version */) {
  uint16_t v;
  a >> v;
  bd = copied_from_poplar::IeeeHalf::fromBits(v);
  ;
}

template <typename Archive>
void serialize(Archive &a,
               copied_from_poplar::IeeeHalf &v,
               const uint32_t version) {
  boost::serialization::split_free(a, v, version);
}

} // namespace serialization
} // namespace boost

namespace poprithms {
namespace compute {
namespace host {

template <typename Archive>
void t_serialize(Archive &a,
                 compute::host::Tensor &td,
                 const uint32_t version) {
  (void)version;
  poprithms::compute::host::Serializer::serialize<Archive>(a, td, version);
}

void BoostSerializer::serialize(boost::archive::text_oarchive &a,
                                poprithms::compute::host::Tensor &b,
                                const uint32_t c) {
  t_serialize(a, b, c);
}

void BoostSerializer::serialize(boost::archive::text_iarchive &a,
                                poprithms::compute::host::Tensor &b,
                                const uint32_t c) {
  t_serialize(a, b, c);
}

template <class Archive>
void t_load_construct_data(Archive &ar,
                           compute::host::Tensor *t,
                           uint32_t v) {
  (void)ar;
  (void)v;
  ::new (t) compute::host::Tensor(
      compute::host::Tensor::scalar(ndarray::DType::Int32, 0));
}

void BoostSerializer::load_construct_data(boost::archive::text_iarchive &a,
                                          poprithms::compute::host::Tensor *b,
                                          uint32_t v) {
  t_load_construct_data(a, b, v);
}

void BoostSerializer::load_construct_data(boost::archive::text_oarchive &a,
                                          poprithms::compute::host::Tensor *b,
                                          uint32_t v) {
  t_load_construct_data(a, b, v);
}

} // namespace host
} // namespace compute
} // namespace poprithms

#define BOOST_CLASS_EXPORT_ETC(T)                                            \
  BOOST_CLASS_EXPORT(poprithms::compute::host::AllocData<T>)                 \
  BOOST_CLASS_EXPORT(poprithms::compute::host::ViewData<T>)                  \
  BOOST_CLASS_EXPORT(poprithms::compute::host::PointerData<T>)               \
  BOOST_SERIALIZATION_ASSUME_ABSTRACT(                                       \
      poprithms::compute::host::OriginData<T>)                               \
  BOOST_SERIALIZATION_ASSUME_ABSTRACT(poprithms::compute::host::TypedData<T>)

BOOST_CLASS_EXPORT_ETC(double)
BOOST_CLASS_EXPORT_ETC(float)
BOOST_CLASS_EXPORT_ETC(int64_t)
BOOST_CLASS_EXPORT_ETC(int32_t)
BOOST_CLASS_EXPORT_ETC(int16_t)
BOOST_CLASS_EXPORT_ETC(int8_t)
BOOST_CLASS_EXPORT_ETC(uint64_t)
BOOST_CLASS_EXPORT_ETC(uint32_t)
BOOST_CLASS_EXPORT_ETC(uint16_t)
BOOST_CLASS_EXPORT_ETC(uint8_t)

BOOST_CLASS_EXPORT_ETC(copied_from_poplar::IeeeHalf)
BOOST_CLASS_EXPORT_ETC(bool)

BOOST_SERIALIZATION_ASSUME_ABSTRACT(poprithms::compute::host::BaseData)
