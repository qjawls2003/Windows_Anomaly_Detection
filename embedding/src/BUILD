load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@python_deps//:requirements.bzl", "all_requirements", "requirement")

py_library(
    name = "tokenizer",
    srcs = [
        "tokenizer.py"
    ],
    deps = [
        requirement("tokenizers"),
    ] #+ all_requirements
    ,
    data = ["//embedding/data:placeholder.txt"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "transformer",
    srcs = [
        "transformer.py"
    ],
    deps = [
        
        requirement("transformers"),
        requirement("filelock"),
        requirement("regex"),
        requirement("huggingface-hub"),
        requirement("packaging"),
        requirement("fsspec"),
        requirement("typing-extensions"),
        requirement("PyYAML"),
        requirement("requests"),
        requirement("urllib3"),
        requirement("certifi"),
        requirement("tqdm"),
        requirement("charset-normalizer"),
        requirement("idna"),
        
    ] #+ all_requirements
    ,
    visibility = ["//visibility:public"],
)

