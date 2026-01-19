# 欢迎访问LOOM-Train使用文档！

LOOM-Train 是一个轻量级、LLM长上下文训练框架。

## 支持

* 
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## 项目布局
    loomtrain
        - core      # 可以使用 core 中的基础类自定义训练逻辑，只需继承LoomTrainer 实现几个工具方法即可。
        - dataset   # 
        - help
        - modeling
        - optim
        - strategy
        - trainer

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
