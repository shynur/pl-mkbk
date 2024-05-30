#! python3.12
import operator
import itertools
import sys
from abc import abstractmethod
from types import NoneType
from typing import Any, Dict, Iterable, List, Self, Tuple, Union, Optional
import lark  # 1.1.9
from enum import Enum


class TermColor(Enum):
    Black = "\033[30m"
    Red = "\033[31m"
    Green = "\033[32m"
    Yellow = "\033[33m"
    Blue = "\033[34m"
    Magenta = "\033[35m"
    Cyan = "\033[36m"
    White = "\033[37m"
    End = "\033[0m"


class PL_Env:
    def __init__(
        self,
        current_env: Optional[Dict[str, Any]] = None,
        *,
        outer: Optional[Self] = None,
    ):
        self.outer: Optional[PL_Env] = outer
        self.current: Dict[
            str, Union[NoneType, int, bool, list, str, "function", PL_LVal]
        ] = ({} if current_env is None else current_env)

    def __getitem__(self, name: str):
        try:
            return self.current[name]
        except KeyError as e:
            if self.outer:
                return self.outer[name]
            else:
                e.add_note("PL: " + f"{name=} doesn’t exist!")
                raise

    def __setitem__(self, name: str, val):
        if name in self.current:
            self.current[name] = val
        else:
            if self.outer:
                self.outer[name] = val
            else:
                raise NameError("PL: " + f"{name=} doesn’t exist!")

    def __repr__(self) -> str:
        return repr(self.outer) + " <- " + repr(self.current)


class PL_LVal:
    @abstractmethod
    def get_val(self) -> Union[NoneType, int, bool, list, str, "function", "PL_LVal"]:
        return self.get_val()

    @abstractmethod
    def set_val(self, val):
        return self.set_val(val)

    val = property(fget=get_val, fset=set_val)


class PL_Var(PL_LVal):
    def __init__(self, *, env: PL_Env, varname: str):
        self.env: PL_Env = env
        self.varname: str = varname

    def get_val(self):
        return self.env[self.varname]

    def set_val(self, val):
        self.env[self.varname] = val


class PL_ArrIdx(PL_LVal):
    def __init__(self, *, arr, idx):
        self.arr = arr
        self.idx = idx

    def get_val(self):
        return self.arr[self.idx]

    def set_val(self, val):
        self.arr[self.idx] = val


def get_val(val):
    if isinstance(val, PL_LVal):
        return val.val
    else:
        return val


class PL_Jumper(Exception):
    def __init__(self, val=None):
        self.val = val


class PL_Breaker(PL_Jumper):
    pass


class PL_Continuer(PL_Jumper):
    pass


class PL_Returner(PL_Jumper):
    pass


class PL_Lambda:
    def __init__(
        self,
        *,
        env: PL_Env,
        parameters: Tuple[str, ...],
        body: Tuple[lark.lexer.Token | lark.tree.Tree, ...],
    ):
        self._env = env
        self._parameters, self._body = parameters, body

    @property
    def env(self):
        return self._env

    @property
    def body(self):
        return self._body

    @property
    def parameters(self):
        return self._parameters

    def __call__(self, *args):
        env = [PL_Env(dict(zip(self.parameters, args)), outer=self.env)]
        try:
            ret = eval_block(self.body, env=env)
        except PL_Returner as r:
            ret = r.val
        except PL_Jumper as j:
            raise PL_Jumper(f"The jumper carries value {j.val}")

        return ret


class PL_Interpreter:
    def __init__(self, *, prelude: Optional[str] = None, debug: bool = False):
        self.debug = debug

        global_environment = PL_Env(
            {  # Functions:
                "Get": lambda varname: (
                    PL_Var(env=global_environment, varname=varname),
                    global_environment.current.setdefault(varname, None),
                )[0],
                "Py": eval,
                # Variables:
                "True": True,
                "False": False,
            }
        )
        self.env = [global_environment]

        if prelude is not None:
            self.run_prog(prelude)

    def start_repl(self):
        parse = lark.Lark(PL_grammar, start="expr_or_stmt").parse
        ps1, ps2 = (
            f"{TermColor.Magenta.value}=>{TermColor.End.value} ",
            f"{TermColor.Green.value}->{TermColor.End.value} ",
        )

        while True:

            while ((txt := input(ps1)) + " ").isspace():
                pass

            while True:
                try:
                    ast = parse(txt)
                    expr_or_stmt: str = ast.data
                    expr = ast.children[0]
                except lark.exceptions.UnexpectedEOF:
                    txt += "\n" + input(ps2)
                except Exception as e:
                    print("Error:", e, file=sys.stderr)
                    if hasattr(e, "__notes__"):
                        print(*e.__notes__, sep="\n", file=sys.stderr)
                    break
                else:
                    if self.debug:
                        print(
                            f"{ast.pretty()}\n{ast}" if hasattr(ast, "pretty") else ast
                        )
                    else:
                        self.exec_stmt(
                            expr,
                            interactive=True,
                            expr_or_stmt=expr_or_stmt,
                        )
                    break

    def run_prog(self, prog: str, /):
        for stmt in lark.Lark(PL_grammar, start="prog").parse(prog).children:
            self.exec_stmt(stmt)

    def exec_stmt(
        self,
        stmt: lark.lexer.Token | lark.tree.Tree,
        /,
        *,
        interactive: bool = False,
        expr_or_stmt: str = "stmt",  # when using REPL user may enter an expression whose value’ll be printed.
    ):
        try:
            ret = get_val(PL_eval(stmt, env=self.env))
        except PL_Jumper as j:
            print("PL:", "Error Jump Statement", j, file=sys.stderr)
            return
        except Exception as e:
            if interactive:
                print("PL:", "Error", e, file=sys.stderr)
                if hasattr(e, "__notes__"):
                    print(*e.__notes__, sep="\n", file=sys.stderr)
            else:
                raise
        except KeyboardInterrupt:
            print("KeyboardInterrupt", file=sys.stderr)
        else:
            if interactive and expr_or_stmt == "expr":
                try:
                    print(f"{TermColor.Cyan.value}{ret}{TermColor.End.value}")
                except Exception as e:
                    print("Error:", e, file=sys.stderr)


def PL_eval(expr: lark.lexer.Token | lark.tree.Tree, *, env: List[PL_Env]):
    if isinstance(expr, lark.lexer.Token):
        if expr.type == "IDENT":
            return PL_Var(env=env[0], varname=expr.value)
        else:
            repr = expr.value
            match expr.type:
                case "STRING":
                    return eval(repr)
                case "INT":
                    return int(repr)
                case unknown_t:
                    raise SyntaxError(
                        "PL: " + f"unknown literal type: ({unknown_t}){expr}!"
                    )

    form = expr.children
    match expr.data if isinstance(expr.data, str) else expr.data.value:
        case "sum_expr" | "product_expr" | "pow_expr":
            return eval_arithm_expr_2op(form, env=env)
        case "not_expr":
            return not get_val(PL_eval(form[0], env=env))
        case "neg_expr":
            return -get_val(PL_eval(form[0], env=env))
        case "var_decl":
            return eval_defvar(form, env=env)
        case "cmp_expr":
            return eval_cmp_expr(form, env=env)
        case "and_expr":
            return eval_and_expr(form, env=env)
        case "or_expr":
            return eval_or_expr(form, env=env)
        case "post_inc_expr" | "post_dec_expr" as expr_t:
            return eval_post_iop_expr(
                form, env=env, op="iadd", val=1 if "inc" in expr_t else -1
            )
        case "pre_inc_expr" | "pre_dec_expr" as expr_t:
            return eval_iop_expr(
                form, env=env, op="iadd", val=1 if "inc" in expr_t else -1
            )
        case "arr_lit":
            return eval_arr_lit(form, env=env)
        case "subscripting":
            return eval_subscripting(form[0], form[1], env=env)
        case "member_accessing":
            field = form[1]
            assert isinstance(field, lark.lexer.Token)
            return eval_subscripting(
                form[0], field.update("STRING", f'"{field.value}"'), env=env
            )
        case "fn_lit":
            varnames, block = form
            return PL_Lambda(
                env=env[0],
                parameters=tuple(
                    getattr(ident, "value") for ident in varnames.children
                ),
                body=tuple(getattr(block, "children")),
            )
        case "funcall":
            fn_expr, args = form
            return eval_funcall(fn_expr, args.children, env=env)
        case "deref_expr":
            return eval_funcall(form[0], [], env=env)
        case "addr_of_expr":
            return lambda lval=PL_eval(form[0], env=env): lval
        case "jmp_expr":
            return eval_jmp(type=getattr(form[0], "value"), expr=form[1], env=env)
        case "assign_expr":
            return eval_assign_expr(*form, env=env)
        case "if_stmt" | "if_expr":
            return eval_ifelse(
                if_=form[0], then_=form[1].children, else_=form[2:], env=env
            )
        case "loop_stmt" | "loop_expr":
            return eval_loop(head=form[:3], body=form[3].children, env=env)
        case "block_expr" | "block_stmt":
            return eval_block(stmts=form, env=env)


def eval_block(
    stmts: Iterable[lark.lexer.Token | lark.tree.Tree],
    *,
    env: List[PL_Env],
):
    ret, env = None, [PL_Env(outer=env[0])]
    for stmt in stmts:
        ret = PL_eval(stmt, env=env)
    return ret


def eval_loop(
    head: List[Optional[lark.lexer.Token | lark.tree.Tree]],
    body: List[lark.lexer.Token | lark.tree.Tree],
    *,
    env: List[PL_Env],
):
    ret, env = None, [PL_Env(outer=env[0])]
    init_clause, cond_expr, iter_expr = head

    if init_clause is not None:
        PL_eval(init_clause, env=env)

    assert cond_expr is not None
    while True:
        iter_env = [PL_Env(outer=env[0])]
        if not get_val(PL_eval(cond_expr, env=iter_env)):
            break

        try:
            ret = eval_block(body, env=iter_env)
        except PL_Continuer as c:
            ret = c.val
        except PL_Breaker as b:
            ret = b.val
            break

        if iter_expr is not None:
            PL_eval(iter_expr, env=iter_env)

    return ret


def eval_ifelse(
    *,
    if_: lark.lexer.Token | lark.tree.Tree,
    then_: List[lark.lexer.Token | lark.tree.Tree],
    else_: List[lark.lexer.Token | lark.tree.Tree],
    env: List[PL_Env],
):
    env = [PL_Env(outer=env[0])]
    return eval_block(then_ if get_val(PL_eval(if_, env=env)) else else_, env=env)


def eval_assign_expr(
    l_expr: lark.lexer.Token | lark.tree.Tree,
    r_expr: lark.lexer.Token | lark.tree.Tree,
    *,
    env: List[PL_Env],
):
    lval = PL_eval(l_expr, env=env)
    assert isinstance(lval, PL_LVal)
    lval.val = get_val(PL_eval(r_expr, env=env))
    return lval


def eval_jmp(
    *,
    expr: Optional[lark.lexer.Token | lark.tree.Tree],
    type: str,
    env: List[PL_Env],
):
    raise {"RETURN": PL_Returner, "BREAK": PL_Breaker, "CONTINUE": PL_Continuer}[type](
        PL_eval(expr, env=env) if expr is not None else None
    )


def eval_funcall(
    fn_expr: lark.lexer.Token | lark.tree.Tree,
    args: List[lark.lexer.Token | lark.tree.Tree],
    /,
    *,
    env: List[PL_Env],
):
    fn = get_val(PL_eval(fn_expr, env=env))
    assert callable(fn)
    return fn(*(get_val(PL_eval(arg, env=env)) for arg in args))


def eval_subscripting(
    arr: lark.lexer.Token | lark.tree.Tree,
    idx: lark.lexer.Token | lark.tree.Tree,
    *,
    env: List[PL_Env],
):
    return PL_ArrIdx(
        arr=get_val(PL_eval(arr, env=env)), idx=get_val(PL_eval(idx, env=env))
    )


def eval_arr_lit(form: List[lark.lexer.Token | lark.tree.Tree], *, env: List[PL_Env]):
    return [get_val(PL_eval(expr, env=env)) for expr in form]


def eval_iop_expr(
    form: List[lark.lexer.Token | lark.tree.Tree], *, env: List[PL_Env], op: str, val
):
    lval = PL_eval(form[0], env=env)
    assert isinstance(lval, PL_LVal)
    lval.val = {"iadd": operator.iadd}[op](lval.val, val)
    return lval


def eval_post_iop_expr(
    form: List[lark.lexer.Token | lark.tree.Tree],
    *,
    env: List[PL_Env],
    val,
    op: str,
):
    lval = PL_eval(form[0], env=env)
    assert isinstance(lval, PL_LVal)
    old_val = get_val(lval)
    lval.val = {"iadd": operator.iadd}[op](lval.val, val)
    return old_val


def eval_or_expr(form: List[lark.lexer.Token | lark.tree.Tree], *, env: List[PL_Env]):
    expr1, expr2 = form
    maybe_lval1 = PL_eval(expr1, env=env)
    if get_val(maybe_lval1):
        return maybe_lval1
    else:
        return PL_eval(expr2, env=env)


def eval_and_expr(form: List[lark.lexer.Token | lark.tree.Tree], *, env: List[PL_Env]):
    expr1, expr2 = form
    maybe_lval1 = PL_eval(expr1, env=env)
    if get_val(maybe_lval1):
        return PL_eval(expr2, env=env)
    else:
        return maybe_lval1


def eval_defvar(form: List[lark.lexer.Token | lark.tree.Tree], *, env: List[PL_Env]):
    ident, expr = form
    assert isinstance(ident, lark.lexer.Token)
    varname, val = ident.value, (
        get_val(PL_eval(expr, env=env)) if expr is not None else None
    )
    env[0] = (new_env := PL_Env({varname: val}, outer=env[0]))
    return PL_Var(env=new_env, varname=varname)


def eval_cmp_expr(form: List[lark.lexer.Token | lark.tree.Tree], *, env: List[PL_Env]):
    a = get_val(PL_eval(form[0], env=env))
    for op_sign, b_expr in itertools.batched(form[1:], 2):
        b = get_val(PL_eval(b_expr, env=env))
        assert isinstance(op_sign, lark.lexer.Token)
        if {
            "EQ_SIGN": operator.eq,
            "NE_SIGN": operator.ne,
            "LE_SIGN": operator.le,
            "GE_SIGN": operator.ge,
            "LT_SIGN": operator.lt,
            "GT_SIGN": operator.gt,
        }[op_sign.type](a, b):
            a = b
        else:
            return False
    return True


def eval_arithm_expr_2op(
    form: List[lark.lexer.Token | lark.tree.Tree], *, env: List[PL_Env]
):
    expr1, op_sign, expr2 = form
    assert isinstance(op_sign, lark.lexer.Token)
    return {
        "PLUS_SIGN": operator.add,
        "MINUS_SIGN": operator.sub,
        "MUL_SIGN": operator.mul,
        "DIV_SIGN": lambda a, b: a // b if type(a) == type(b) == int else a / b,
        "POW_SIGN": operator.pow,
    }[op_sign.type](get_val(PL_eval(expr1, env=env)), get_val(PL_eval(expr2, env=env)))


PL_grammar: str = """
# Architecture
prog: stmt*
?stmt: expr ";"
     | block_stmt
     | if_stmt | loop_stmt

# Expression
## Pre-Requisite
?or_expr: and_expr
        | or_expr "OR" and_expr
?and_expr: not_expr
         | and_expr "AND" not_expr
!?not_expr: cmp_expr
          | "NOT" not_expr
?sum_expr: product_expr
         | sum_expr (PLUS_SIGN|MINUS_SIGN) product_expr
?product_expr: neg_like_expr
             | product_expr (MUL_SIGN|DIV_SIGN) neg_like_expr
?neg_like_expr: pow_expr
              | "-"  neg_like_expr -> neg_expr
              | "++" neg_like_expr -> pre_inc_expr
              | "--" neg_like_expr -> pre_dec_expr
              | "&"  neg_like_expr -> addr_of_expr
              | "*"  neg_like_expr -> deref_expr
?pow_expr: funcall_like_expr
         | funcall_like_expr POW_SIGN neg_like_expr
?funcall_like_expr: atom_expr
                  | subscripting | funcall
                  | atom_expr "." CNAME -> member_accessing
                  | funcall_like_expr "++" -> post_inc_expr
                  | funcall_like_expr "--" -> post_dec_expr
## Main
?expr: assign_expr
     | (CONTINUE|BREAK|RETURN) [expr] -> jmp_expr
     | "LET" IDENT ["=" expr]         -> var_decl
?assign_expr: logic_expr
            | logic_expr "=" assign_expr
?logic_expr: or_expr
?cmp_expr: sum_expr ((EQ_SIGN|NE_SIGN|LE_SIGN|GE_SIGN|LT_SIGN|GT_SIGN) sum_expr)*
?atom_expr: IDENT
          | lit_val | "(" expr ")" | block_expr
          | if_expr | loop_expr

# Array
arr_lit: "[" (expr ("," expr)*)? "]"
subscripting: funcall_like_expr "[" expr "]"

# Function
fn_lit: parameters (block_stmt|block_expr)
funcall: funcall_like_expr args

# Statements
if_stmt: "IF" expr block_stmt ("ELSE" (if_stmt | "{" stmt*     "}"))?
if_expr: "IF" expr block_expr ("ELSE" (if_expr | "{" stmt* expr"}"))?

loop_stmt: ("FOR"|"WHILE") _loop_head block_stmt
loop_expr: ("FOR"|"WHILE") _loop_head block_expr
_loop_head: [UNMATCHABLE]  expr      [UNMATCHABLE]
          |    [expr] ";"  expr ";"     [expr]

# Misc
CONTINUE: "CONTINUE"
BREAK: "BREAK"
RETURN: "RETURN"
IDENT: (LETTER|"_")+ (LCASE_LETTER|"_"|DIGIT) (LETTER|"_"|DIGIT)*
     |               (LCASE_LETTER|"_")       (LETTER|"_"|DIGIT)*
parameters: "|" (IDENT ("," IDENT)*)? "|"
block_stmt: "{" stmt*      "}"
block_expr: "{" stmt* expr "}"
args: "(" (expr ("," expr)*)? ")"
?lit_val: INT
        | arr_lit
        | fn_lit
        | STRING
POW_SIGN: "^"
MUL_SIGN: "*"
DIV_SIGN: "/"
PLUS_SIGN: "+"
MINUS_SIGN: "-"
EQ_SIGN: "=="
NE_SIGN: "!=" | "<>"
LE_SIGN: "<="
GE_SIGN: ">="
LT_SIGN: "<"
GT_SIGN: ">"
UNMATCHABLE: /^o^/
expr_or_stmt: stmt -> stmt
            | expr -> expr

# Also see <~/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/lark/grammars/common.lark>.  (Personal note.)
%import common.ESCAPED_STRING -> STRING
%import common.INT
%import common (LCASE_LETTER, LETTER, DIGIT, CNAME)
%import common (WS, NEWLINE, SH_COMMENT)
%ignore WS
%ignore SH_COMMENT
"""

if __name__ == "__main__":

    # Get Command Line Options and Arguments
    import argparse

    cmdarg_parser = argparse.ArgumentParser(
        prog="PL/Macabaca",
        description="An interpreter for the programming language “PL/Macabaca” designed by shynur.",
        epilog="""
Project URL: <https://github.com/shynur/pl-mkbk>.
Copyright © 2024  Xie Qi.  All rights reserved.
""",
    )
    cmdarg_parser.add_argument(
        "-i",
        action="store_true",
        dest="interactive",
        default=None,
        help="交互模式",
    )
    cmdarg_parser.add_argument("filename", type=str, nargs="?", help="源文件")
    cmdargs = cmdarg_parser.parse_args()

    # Create Interpreter
    i = PL_Interpreter(
        prelude="""
Get("Print"    ) = Py("print ");
Get("Sorted"   ) = Py("sorted");
Get("Min"      ) = Py("min   ");
Get("Max"      ) = Py("max   ");
Get("Scan"     ) = Py("input ");
Get("Len"      ) = Py("len   ");
Get("Sum"      ) = Py("sum   ");
Get("Dict"     ) = Py("dict  ");
Get("Set"      ) = Py("set   ");
Get("Copy"     ) = Py("__import__('copy').copy"    );
Get("DeepCopy" ) = Py("__import__('copy').deepcopy");
Get("Fargs"    ) = Py("lambda f: lambda *args: f([*args])");  # func(List) -> func(*args)
Get("DefStruct") = Py("lambda *fields:  \
    __import__('dataclasses').dataclass(  \
        type('结构体', (), {  \
            '__annotations__': {field: object for field in fields},  \
            '__getitem__': lambda self, field: getattr(self, field),  \
            '__setitem__': lambda self, field, new_val: setattr(self, field, new_val),  \
}))");
""",
    )

    # 1. Execute Source Code
    # 2. Inferior Mode
    if cmdargs.filename is not None:
        i.run_prog(open(cmdargs.filename, encoding="utf-8").read())
        if cmdargs.interactive is None:
            sys.exit()
    i.start_repl()
