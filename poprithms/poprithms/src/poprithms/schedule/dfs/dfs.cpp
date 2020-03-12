#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <poprithms/schedule/dfs/dfs.hpp>
#include <poprithms/schedule/dfs/error.hpp>

namespace poprithms {
namespace schedule {
namespace dfs {

std::vector<uint64_t> postOrder(const Edges &edges) {
  if (edges.size() == 0) {
    return {};
  }

  // Construct Boost Graph
  boost::adjacency_list<> g;
  for (uint64_t i = 0; i < edges.size(); ++i) {
    boost::add_edge(i, i, g);
    for (auto y : edges[i]) {
      boost::add_edge(i, y, g);
    }
  }

  auto nOps = boost::num_vertices(g);
  if (nOps != edges.size()) {
    throw dfs::error(
        "Expected Boost Graph to contain as many Nodes as input Edges");
  }
  constexpr uint64_t NotVisited{std::numeric_limits<uint64_t>::max()};
  std::vector<uint64_t> ftime(nOps, NotVisited);
  std::vector<boost::default_color_type> color_map(nOps);
  int t{-1};
  for (uint64_t i = 0; i < nOps; ++i) {
    if (ftime[i] == NotVisited) {
      boost::depth_first_visit(g,
                               boost::vertex(i, g),
                               boost::make_dfs_visitor(boost::stamp_times(
                                   &ftime[0], t, boost::on_finish_vertex())),
                               boost::make_iterator_property_map(
                                   &color_map[0],
                                   boost::get(boost::vertex_index, g),
                                   color_map[0]));
    }
  }
  return ftime;
}

} // namespace dfs
} // namespace schedule
} // namespace poprithms
