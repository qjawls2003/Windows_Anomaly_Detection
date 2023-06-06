load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@python_deps//:requirements.bzl", "requirement")


py_library(
    name = "preprocess",
    srcs = [
        "preprocess.py"
    ],
    deps = [
        requirement("pandas"),
    ],
    visibility = ["//visibility:public"],
    data = ["//pre_processing/data:WinEvent4688.csv"],
)

