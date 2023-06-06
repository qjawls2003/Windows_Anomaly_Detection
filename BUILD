load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@python_deps//:requirements.bzl", "requirement")


py_binary(
    name = "main",
    srcs = [
        "main.py"
    ],
    deps = [
        "//pre_processing/src:preprocess",
        "//pre_processing/src:vectorize",
        "//pre_processing/src:clustering"
    ],
    
)