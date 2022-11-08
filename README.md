# 暗通道先验去雾算法

## 介绍

实现了暗通道先验去雾算法和使用了导向滤波的暗通道先验去雾算法。

## 安装 ##

- [可选]使用`conda`或`venv`创建虚拟环境

- ```bash
  pip install -r requirments.txt
  ```

## 用法

- ```bash
  python dehaze.py [--input-img 输入图像路径, 默认为 image/example1.png] [--output-path 输出路径, 默认为 output] [--alg dark_channel 或 guided_filter, 默认为 dark_channel，不使用导向滤波，选择 guided_filter 以启用导向滤波]
  ```
