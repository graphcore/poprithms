// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SERIALIZATION_SERIALIZER_HPP
#define POPRITHMS_SERIALIZATION_SERIALIZER_HPP

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>

#include <poprithms/ndarray/shape.hpp>

namespace poprithms {
namespace ndarray {

class Serializer {

public:
  template <typename Archive>
  // Serialize a Shape object
  static void serialize(Archive &a, ndarray::Shape &shape, uint32_t version) {
    (void)version;
    // Access to private member 'shp' granted through friend-ship.
    a &shape.shp;
  }
};

} // namespace ndarray
} // namespace poprithms

namespace boost {
namespace serialization {

using namespace poprithms;

template <typename Archive>
void serialize(Archive &a, ndarray::Shape &t, const uint32_t version) {
  (void)version;
  poprithms::ndarray::Serializer::serialize<Archive>(a, t, version);
}

} // namespace serialization
} // namespace boost

#endif
