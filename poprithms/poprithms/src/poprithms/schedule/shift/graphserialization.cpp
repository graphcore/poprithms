// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <sstream>
#include <string>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <schedule/shift/error.hpp>
#include <schedule/shift/graphserialization.hpp>

#include <poprithms/schedule/shift/logging.hpp>

namespace poprithms {
namespace schedule {
namespace shift {
namespace serialization {

namespace {

template <typename T> T t_stoi(const std::string &x);
template <> uint64_t t_stoi(const std::string &x) { return std::stoull(x); }

template <typename T> T getNumericType(const std::string &x) {
  static_assert(std::is_integral<T>::value, "Integral required.");
  if (x.empty()) {
    std::ostringstream oss;
    oss << "No chars (and therefore no digits) detected in getNumericType("
        << x << ").";
    throw error(oss.str());
  }
  if (!std::all_of(x.cbegin(), x.cend(), [](char i) {
        return std::isdigit(i) || i == '-';
      })) {
    std::ostringstream oss;
    oss << "Not all chars are digits in getNumericType(" << x << ").";
    throw error(oss.str());
  }
  return t_stoi<T>(x);
}

template <> double getNumericType<double>(const std::string &x) {
  return std::stod(x);
}

} // namespace

Graph fromSerializationString(const std::string &serialization) {

  std::istringstream ist(serialization);
  boost::property_tree::ptree tree;

  try {
    log().trace("Entering boost::propery_tree::read_json");
    boost::property_tree::read_json(ist, tree);
  } catch (boost::property_tree::ptree_error &e) {
    throw error(std::string("Failed to parse string to JSON: ") + e.what());
  }

  // Gather Op data from boost data-structure
  std::vector<OpAddress> opAddresses;
  std::vector<std::string> debugStrings;
  std::vector<int64_t> fwdLink_i64s;
  std::vector<std::vector<OpAddress>> outs;
  std::vector<std::vector<AllocAddress>> allocs;
  for (const auto &[k, opEntry] : tree.get_child("ops")) {
    (void)k;
    opAddresses.push_back(opEntry.get<OpAddress>("address"));
    debugStrings.push_back(opEntry.get<std::string>("debugString"));
    fwdLink_i64s.push_back(opEntry.get<int64_t>("fwdLink"));
    outs.push_back({});
    for (auto out : opEntry.get_child("outs")) {
      outs.back().push_back(getNumericType<OpAddress>(out.second.data()));
    }
    allocs.push_back({});
    for (auto alloc : opEntry.get_child("allocs")) {
      allocs.back().push_back(
          getNumericType<AllocAddress>(alloc.second.data()));
    }
  }

  // Gather Alloc data from boost data-structure
  std::vector<AllocAddress> allocAddresses;
  std::vector<std::vector<double>> weights;
  for (const auto &[k, allocEntry] : tree.get_child("allocs")) {
    (void)k;
    allocAddresses.push_back(allocEntry.get<AllocAddress>("address"));
    weights.push_back({});
    for (auto w : allocEntry.get_child("weight")) {
      weights.back().push_back(getNumericType<double>(w.second.data()));
    }
  }

  // Map index in data from boost data-structure to OpAddresses,
  // AllocAddresses
  constexpr auto UnsetIndex = std::numeric_limits<uint64_t>::max();
  std::vector<uint64_t> opToInd(opAddresses.size(), UnsetIndex);
  for (uint64_t i = 0; i < opAddresses.size(); ++i) {
    if (opAddresses[i] >= opAddresses.size() ||
        opToInd[opAddresses[i]] != UnsetIndex) {
      throw error("Invalid OpAddress while parsing JSON : they must be "
                  "unique and less than nOps()");
    }
    opToInd[opAddresses[i]] = i;
  }
  std::vector<uint64_t> allocToInd(allocAddresses.size(), UnsetIndex);
  for (uint64_t i = 0; i < allocAddresses.size(); ++i) {
    if (allocAddresses[i] >= allocAddresses.size() ||
        allocToInd[allocAddresses[i]] != UnsetIndex) {
      throw error("Invalid AllocAddress while parsing JSON : they must be "
                  "unique and less than nOps()");
    }
    allocToInd[allocAddresses[i]] = i;
  }

  log().trace("Constructing Graph from boost::property_tree::ptree");
  // Construct graph
  Graph graph;

  // 1) insert Ops
  for (OpAddress add = 0; add < opAddresses.size(); ++add) {
    auto ind = opToInd[add];
    graph.insertOp(debugStrings[ind]);
  }

  // 2) insert Allocs
  for (AllocAddress add = 0; add < allocAddresses.size(); ++add) {
    auto ind = allocToInd[add];
    if (weights[ind].size() != NAW) {
      throw error("Unexpected number of weight values in parsing JSON");
    }
    std::array<double, NAW> v;
    for (uint64_t i = 0; i < NAW; ++i) {
      v[i] = weights[ind][i];
    }
    graph.insertAlloc(v);
  }

  // 3) insert Links, Constrains, Op-Alloc associations
  for (OpAddress add = 0; add < opAddresses.size(); ++add) {
    auto ind = opToInd[add];
    if (fwdLink_i64s[ind] >= 0 &&
        static_cast<uint64_t>(fwdLink_i64s[ind]) < opAddresses.size()) {
      graph.insertLink(add, static_cast<OpAddress>(fwdLink_i64s[ind]));
    }
    for (auto o : outs[ind]) {
      graph.insertConstraint(add, o);
    }

    for (auto a : allocs[ind]) {
      graph.insertOpAlloc(add, a);
    }
  }
  return graph;
}

} // namespace serialization
} // namespace shift
} // namespace schedule
} // namespace poprithms
