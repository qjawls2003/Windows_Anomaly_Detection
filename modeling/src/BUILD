load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@python_deps//:requirements.bzl", "all_requirements", "requirement")


py_library(
    name = "model",
    srcs = [
        "model.py"
    ],
    deps = [      
        requirement("torch"),
        requirement("torchvision"),
        requirement("torchaudio"),
    ] #+ all_requirements
    ,
    visibility = ["//visibility:public"],
)