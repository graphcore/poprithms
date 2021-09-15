// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/AffineExpr.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <poprithms/error/error.hpp>

int main() {
  // Some silly tests. What we're really interested in here is that this test
  // builds, that we have no problem finding and linking with boost, llvm and
  // mlir.
  llvm::SmallVector<mlir::AffineExprKind> kinds{
      mlir::AffineExprKind::Constant};
  if (kinds.empty()) {
    throw poprithms::test::error("kinds should have 1 element");
  }

  boost::property_tree::ptree tree;
  auto x = std::distance(tree.rbegin(), tree.rend());
  if (x != 0) {
    throw poprithms::test::error("default constructed ptee should be empty");
  }

  return 0;
}
