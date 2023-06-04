load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@python_deps//:requirements.bzl", "requirement")

py_library(
    name = "csv_to_df",
    srcs = [
        "csv_to_df.py"
    ],
    deps = [
    ],
    visibility = ["//visibility:public"]
)

py_binary(
    name = "main",
    srcs = [
        "main.py"
    ],
    deps = [
        ":csv_to_df",
    ]
)