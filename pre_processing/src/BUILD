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

py_library(
    name = "tokenizer",
    srcs = [
        "tokenizer.py"
    ],
    deps = [
        requirement("tokenizers"),
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "clustering",
    srcs = [
        "clustering.py"
    ],
    deps = [
        requirement("scikit-learn"),
    ],
    visibility = ["//visibility:public"],
)