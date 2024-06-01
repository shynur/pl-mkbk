# PL-玛卡巴卡

自己设计的编程语言, 解释器代码在 [pl.py](pl.py) 文件中.  <br>
可直接执行; 键入

```
python pl.py -h
```

以查看帮助信息.

## 语言标准

见 [Specification](Specification.md).

## 解释器食用指南

### 输入输出

解释器接受*表达式*或*语句*.
允许多行输入, 由解释器自行检查是否到达*表达式*或*语句*的末尾.

<img src="https://github.com/shynur/pl-mkbk/assets/98227472/b98e9554-6baf-44c9-b234-95413860a645" width="25%">

对于*表达式*, 解释器会用不同颜色的字体打印出返回值.

### 出错恢复

在等待解释器执行的过程中, 可键入 <kbd>Control-C</kbd> 以终止执行; 解释器会等待新的输入.

若想舍弃当前输入的内容, 请键入 <kbd>Control-Z</kbd> <kbd>Enter</kbd>.

### 退出解释器

狂按 <kbd>Control-C</kbd>.

## License

*目前*未指定许可证.

> Without a license, the default copyright laws apply, meaning that I retain all rights to my source code and no one may reproduce, distribute, or create derivative works from my work.
