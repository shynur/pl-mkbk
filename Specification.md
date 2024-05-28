# 名字

## 变量名

**纯大写字母构成的名字**是*保留字*, 不可用作*变量名*.
除此以外, *变量名*的合法性同 C 语言.

## 内置变量

*内置变量*存在于*全局环境*中, 包括但不限于: `True`, `False`, `Py`, `Get`.
它们皆由 **单个大写字母** + **多个小写字母** 组成.

# 语法

## 变量声明

```
LET var;  # 仅声明, 省略变量的值.
LET var = expr;
LET var: str;  # 可选的类型注解
LET var: int = 0;
```

(未赋值的变量 的 值 是不可预知的!)

## 条件分支

```
IF 1 {2;}
IF 1 {2;} ELSE {3;}
IF 1 {2;} ELSE IF 3 {4;}
IF 1 {2;} ELSE IF 3 {4;} ELSE {5;}
```

## 循环

`FOR` 与 `WHILE` 是**同义词**.

循环由 循环头 和 循环体 组成, 循环头 有以下形式:

```
FOR b {}      FOR ;b {}     FOR ;b; {}
FOR a;b {}    FOR a;b; {}
FOR a;b;c {}
```

其中, ‘a’, ‘b’, ‘c’ 的含义类似 C 语言中的: ‘for(a;b;c){}’.



# API

## `True` and `False`

它们只是*全局变量*, 因此可以被*覆盖*.

例:

```
True = False;
True;      # => 打印出‘False’
NOT True;  # => 打印出‘True’
```

## `Get`

从*全局环境*中获取变量 (的左值).

例:

```
LET True: int = 0;  # 覆盖原本的‘True’
True;               # => 打印出‘0’
Get("True");        # => 打印出‘True’
```

## `Py`

调用 Python 中的 `eval` 函数.
