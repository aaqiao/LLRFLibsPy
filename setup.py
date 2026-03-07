from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name              = "llrflibs",
      version           = "1.0.2",
      maintainer        = "Zheqiao Geng",
      maintainer_email  = "zheqiao.geng@psi.ch",
      author            = "Zheqiao Geng",
      author_email      = "zheqiao.geng@psi.ch",
      description       = "LLRF algorithm libraries in Python.",
      long_description  = long_description,
      long_description_content_type = "text/markdown",
      url               = "https://github.com/aaqiao/LLRFLibsPy",
      license           = "BSD 3-Clause",
      packages          = ['llrflibs'],
      zip_safe          = False)
