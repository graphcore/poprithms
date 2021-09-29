// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <poprithms/error/error.hpp>

int main() {

  boost::property_tree::ptree tree;
  auto x = std::distance(tree.rbegin(), tree.rend());
  if (x != 0) {
    throw poprithms::test::error("default constructed ptee should be empty");
  }

  return 0;
}
