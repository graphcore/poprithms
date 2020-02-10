# Bazel interface for building poprithms with Tensorflow
licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# Target for the whole of the poprithms library
cc_library(
    name = "poprithms",
    srcs = glob(["src/**/*.cpp"]),
    hdrs = glob(["include/**/*.hpp"]),
    strip_include_prefix = "include",
)

