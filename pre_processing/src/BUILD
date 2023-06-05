load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@python_deps//:requirements.bzl", "requirement")


py_library(
    name = "csv_to_df",
    srcs = [
        "csv_to_df.py"
    ],
    deps = [
        requirement("pandas"),
    ],
    visibility = ["//visibility:public"],
    data = ["//pre_processing/data:WinEvent4688.csv"],
)

py_library(
    name = "vectorize",
    srcs = [
        "vectorize.py"
    ],
    deps = [
        requirement("gensim"),
    ],
    visibility = ["//visibility:public"],
)