# 名字

## 变量名

所有**纯大写字母构成的名字**是*保留字*, 不可用作*标识符*.
除此以外, *标识符*的合法性同 C 语言.

## 内置变量

*内置变量*存在于*全局环境*中, 包括但不限于: `True`, `False`, `Py`, `Get`.
它们皆以 **单个大写字母** 打头.

# 作用域

## 词法作用域

### 变量声明

变量必须先**声明** (可以不赋值, 但程序不应依赖未赋值的变量的值!), 方可使用.
被声明的变量会*遮蔽 (shadow)* 活跃的同名变量, 这一点与 Rust 一致.
例如,

```
LET n = 0;
LET n = n + 1;  # 先执行右侧表达式.  此时该行的‘LET’后的第一个‘n’还未被声明,
                # 所以第一行的‘n’仍处于活跃状态.
Print(n);       # => 打印‘1’
```

### 闭包

闭包会捕获环境中的*自由变量*, 且**指向性保证不变**, 这一点与 Go 一致.
它的语法很像 Ruby, 例如,

```
|x| {
    LET f = || {Print(x);};  # 此时第一个‘x’活跃, 所以它被捕获.
    f();
    LET x = 2;  # 虽然对‘x’重新声明, 但闭包捕获的仍然是被遮蔽的那个.
    f  # 函数默认返回最后一个表达式的值, 比如这里的‘f’.
}(1)();
```

#### 定义递归函数的快捷方式

显然以下尝试定义阶乘函数的代码是失败的:

```
LET fact = |n| {
    IF n==0 {
        1
    } ELSE {
        # 当该函数被创建时,
        # 并没有‘fact’处于活跃状态!
        n * fact(n-1)
    }
};
fact(5);  # => Error!
```

我们应该分离*变量声明*和*赋值*:

```
LET fact;
fact = |n| {IF n==0 {1} ELSE {n*fact(n-2)}};
```

这么写有些繁琐, 但也是 Go 语言官方文档推荐的解决方案; Rust 的写法可能更加难读.
所幸我们的编程语言是**基于表达式**的, 声明语句也不过是声明表达式添上了分号.
因此,

```
(LET fact) = |n| {
    IF n==0 {
        1
    } ELSE {
        n * fact(n-1)
    }
};
Print(fact(5));  # => 打印‘120’
```

此处**‘=’(赋值运算符) 左侧*表达式*被先执行** (但‘=’仍然是右结合的):
1. 词法变量‘fact’被声明, 此时它进入活跃状态;
2. ‘LET’表达式返回了‘fact’的*左值*;
3. 执行右侧表达式, 即创建匿名函数 (闭包).  它被创建时, 捕获了环境中活跃的‘fact’.
4. ‘fact’被赋值了右侧创建的闭包.

##### Y 组合子

本节给出一种在匿名函数中使用递归的技巧:

```
LET fact = |f| {
    |g| { g(g) } (
        |x| {
            f(|arg| { x(x)(arg) })
        }
    )
}(|f| {
    |n| {
        IF n==0 {
            1
        } ELSE {
            n * f(n-1)
        }
    }
});
Print(fact(5));  # => 打印‘120’
```

可以看到, 在定义‘fact’时, ‘fact’的函数体内并未引用‘fact’.

## 动态作用域

语言实现了对动态作用域的*有限支持*.

从解释器准备执行第一条语句时, 词法作用域的规则就开始且永久生效了.
但在词法作用域之外 (也就是所有代码执行前), 有一个隐藏的全局作用域.

全局作用域中的变量, 会被词法变量的声明所遮蔽, 但是有办法在代码中的任何位置访问.  <br>
就像 Emacs Lisp (或 Common Lisp) 那样, 动态绑定的变量通常 (且只能, 对于本语言来说) 在*全局作用域*中声明.
我们用 `Get` 函数来引用全局变量:

```
Get("x") = 0;  # 在全局作用域中创建变量‘x’.
LET print_x = || { Print(Get("x")); };  # 打印动态绑定的变量‘x’的值.
{
    LET x = Get("x");  # 保存‘x’的旧值.
    Get("x") = 1;  # ‘x’被重新绑定了.
    print_x();     # => 打印‘1’
    Get("x") = x;  # ‘x’被恢复绑定了.
}
print_x();  # => 打印‘0’
```

将*动态作用域*的完整支持加入到语言本身当中很轻松就可以实现, 但没有必要这么做, 使用标准库的 `Get` 函数足矣.

# 语句

*语句*由*表达式*添上分号 (`;`) 构成;
此外, *语句块*也是*语句*, 例如,

```
{}
{ 1; }
```

## 表达式

***表达式*会返回值**, 例如,

```
2^-1     # => 0.5
"hello"  # => "hello"
[1,2+3]  # => [1,5]
True     # => 逻辑真
||{}     # => 闭包
```

以*表达式*结尾的代码块也有返回值 (由末尾的表达式产生), 它也是*表达式*.
例如,

```
{ 1; 2 }  # => 2
```

*语句* 或 没有返回值的*表达式* (例如 `||{}()`, 调用函数体为空的闭包) *通常*返回‘None’作为占位符, 用户不应依赖这一行为.
因为语言未提供直接访问‘None’的名字, 正如初代 JavaScript, 且本意如此.

## 作用域

代码块会新建一个词法作用域, 这和 C++ 是一样的.
例如,

```
LET x = 0;
{ LET x = 1 };
{ LET x = 1; }
Print(x);  # => 打印‘0’
```

# 控制流

## 条件分支

以下是语句:

```
IF 1 {2;} ELSE IF 3 {4;} ELSE {5;}
IF 1 {2 } ELSE IF 3 {4 } ELSE {5};
```

以下是表达式:

```
IF 1 {
    2
} ELSE IF 3 {
    4
} ELSE {
    5
}
```

可以有任意多个 `ELSE IF` 从句; 收尾的 `ELSE` 是可选的.

### `IF` 的作用域

在 `IF` 的条件表达式中所声明的变量 的 作用域截止于整个 `IF` 语句的结束.
以下语句合法:

```
IF LET a = False {
    Print(a);
} ELSE IF LET b = False {
    Print(a, b);
} ELSE {
    Print(a, b);
}
```

注意, `IF` 的 then-从句 和 else-从句 都是代码块.

## 循环

`FOR` 与 `WHILE` 是**同义词**.

循环由 循环头 和 循环体 (代码块) 组成, 有以下两种形式:

```
FOR a; b; c {}
FOR b {}  # 源自 Go 语言, 仅仅是上一种写法的略写.
```

其中 ‘a’,‘b’,‘c’ 的含义及其中声明的变量的作用域范围与 C++ 标准所定义的一致;

### 跳转指令

`BREAK [expr]` 和 `CONTINUE [expr]` **表达式** 的效果与 C 所定义的差不多, 只是它们还可以携带值.
例如,

```
FOR LET i=0; (LET j=i)<=3; i++ {
    IF j==3 {
        CONTINUE j  # 但是允许 ‘CONTINUE’ 携带值 似乎没什么用?
    }
}  # => 3
```

注意, 上述代码中使用了 `(LET j=i)<=3`.
这是因为‘i’的作用域贯穿整个循环, 而‘j’截止于代码块的结束.  <br>
该语言会**尽可能地传递*左值***, 因此若改成 `CONTINUE i` 则最终返回 `4`.

# API

## `True` and `False`

它们只是*全局变量*, 因此可以被*覆盖*.

例:

```
True = False;
Print(True);   # => 打印出‘逻辑假’
Print(NOT 0);  # => 打印出‘逻辑真’
```

## `Get`

从*全局环境*中获取变量 (的*左值*).

例:

```
LET True;            # 覆盖原本的‘True’
Print(Get("True"));  # => 打印出‘逻辑真’
```

## `Py`

调用 Python 中的 `eval` 函数.

例:

```
Py("set")([1,2,3]).__len__()  # => 3
```

可以看出, 该语言与 Python 有极好的交互性.

## 非必需 API

### `Fargs`: 传递任意数量的参数

语言本身不支持任意元参数, 但

```
LET print_each = Fargs(|args| {
    FOR LET i = 0; i < Len(args); i++ {
        Print(args[i])
    }
});
print_each(1, 2, 3);
```

### `Struct`: 定义结构体

```
LET Student = Struct("name", "age");
LET shynur = Student("XQ", 21);
Print(shynur);
shynur.name = "xq";
shynur["age"] = 22;  # 下标查询
Print(shynur);
```

在访问对象属性这一点上, 它表现得很像 JavaScript.

# 运算符

## 地址

支持 C 语义的*取址*与*解引用*.

```
LET arr = 0;
LET p = &arr;
*p = [1, [2]];
*&**&&**&**&&&arr[1][0] = 3;
Print(arr)  # => 打印‘[1, [3]]’
```

对于 `struct.field` 的表示方式同理, 其 `field` 也有地址.

更一般地, 所有值都支持*取址*与*解引用*, 但只有*左值*可以被赋值.  <br>
以下代码展示了该语言如何使用闭包实现*取址*与*解引用*的语法糖:

```
LET x = 0;
LET p = ||{ x };
p()       # => 返回‘0’
p() = 1;
x         # => 返回‘1’
```

## 关系运算符

使用短路求值.
支持 `1 < 2 < 3` 的连续比较的写法.

表示‘不相等’允许使用 C++ 的 `operator!=` 写法 和 SQL 的 `<>` 写法.

## 布尔运算

使用短路求值, 优先级同 Python.