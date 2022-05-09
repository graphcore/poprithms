// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_NDARRAY_SERIALIZER_HPP
#define POPRITHMS_NDARRAY_SERIALIZER_HPP

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace ndarray {

class BoostSerializer {

public:
  template <typename Archive>
  // Serialize a Shape object
  static void serialize(Archive &a, ndarray::Shape &shape, uint32_t version);
};

} // namespace ndarray
} // namespace poprithms

namespace boost {
namespace serialization {

using namespace poprithms;

void serialize(boost::archive::text_iarchive &a,
               ndarray::Shape &t,
               const uint32_t version) {
  (void)version;
  poprithms::ndarray::BoostSerializer::serialize(a, t, version);
}

void serialize(boost::archive::text_oarchive &a,
               ndarray::Shape &t,
               const uint32_t version) {
  (void)version;
  poprithms::ndarray::BoostSerializer::serialize(a, t, version);
}

} // namespace serialization
} // namespace boost

#endif
