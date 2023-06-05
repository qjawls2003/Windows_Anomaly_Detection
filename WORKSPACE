load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    sha256 = "94750828b18044533e98a129003b6a68001204038dc4749f40b195b24c38f49f",
    strip_prefix = "rules_python-0.21.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.21.0/rules_python-0.21.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

python_register_toolchains(
    name = "python3_8",
    python_version = "3.8",
)

load("@python3_8//:defs.bzl", "interpreter")

load("@rules_python//python:pip.bzl", "pip_parse")
 
pip_parse(
   name = "python_deps",
   requirements_lock = "//:requirements_lock.txt",
   python_interpreter_target = interpreter
)

load("@python_deps//:requirements.bzl", "install_deps")

install_deps()