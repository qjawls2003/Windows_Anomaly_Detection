load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@python_deps//:requirements.bzl", "requirement")

py_library(
    name = "tokenizer",
    srcs = [
        "tokenizer.py"
    ],
    deps = [
        requirement("tokenizers"),
    ],
    data = ["//embedding/data:placeholder.txt"],
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