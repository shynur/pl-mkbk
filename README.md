# PL/玛卡巴卡

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

在等待解释器执行的过程中, 可键入 <kbd>Control-C</kbd> 以终止执行;  \
若想舍弃当前输入的内容, 请键入 <kbd>Control-C</kbd>.

总而言之, <kbd>Control-C</kbd> 会带你重回顶层, 并等待新的输入.

### 退出解释器

只要 输入文本中 任一行的行首字符是 `^Z` (由 <kbd>Control-Z</kbd> 产生), 解释器就会退出.
例如,

```
=> 1 +
-> ^Z 233
EOFError
解释器退出了...
```

## License

See the [LICENSE.md](./LICENSE.md) file for license rights and limitations (LGPLv3).

_____

Copyright &copy; 2024  [*谢骐*](https://github.com/shynur).  All rights reserved.
