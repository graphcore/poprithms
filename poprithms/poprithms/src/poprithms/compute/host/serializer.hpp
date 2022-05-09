// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_SERIALIZER_HPP
#define POPRITHMS_COMPUTE_HOST_SERIALIZER_HPP

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/operators.hpp>

#include <poprithms/compute/host/tensor.hpp>

namespace poprithms {
namespace compute {
namespace host {

// The design pattern used for the serialization is unfortunately very
// verbose, but the "cleaner" way of implementing these methods directly in
// the boost::serialization namespace caused linker errors which I could not
// resolve.
class BoostSerializer {
private:
public:
  static void
  serialize(boost::archive::text_oarchive &, Tensor &, uint32_t version);

  static void
  serialize(boost::archive::text_iarchive &, Tensor &, uint32_t version);

  static void load_construct_data(boost::archive::text_iarchive &a,
                                  poprithms::compute::host::Tensor *b,
                                  uint32_t v);

  static void load_construct_data(boost::archive::text_oarchive &a,
                                  poprithms::compute::host::Tensor *b,
                                  uint32_t v);
};

} // namespace host
} // namespace compute
} // namespace poprithms

// These are the template overrides in the boost::serialization namespace
// which are ultimately used for serialization.
namespace boost {
namespace serialization {

void serialize(boost::archive::text_oarchive &a,
               poprithms::compute::host::Tensor &b,
               uint32_t v) {
  poprithms::compute::host::BoostSerializer::serialize(a, b, v);
}

void serialize(boost::archive::text_iarchive &a,
               poprithms::compute::host::Tensor &b,
               uint32_t v) {
  poprithms::compute::host::BoostSerializer::serialize(a, b, v);
}

// The host::Tensor class does not have a default constructor, so following
// the advice in the boost docs to handle this case.
void load_construct_data(boost::archive::text_oarchive &a,
                         poprithms::compute::host::Tensor *b,
                         uint32_t v) {
  poprithms::compute::host::BoostSerializer::load_construct_data(a, b, v);
}

void load_construct_data(boost::archive::text_iarchive &a,
                         poprithms::compute::host::Tensor *b,
                         uint32_t v) {
  poprithms::compute::host::BoostSerializer::load_construct_data(a, b, v);
}

} // namespace serialization
} // namespace boost

#endif
