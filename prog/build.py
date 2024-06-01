#! python3.12
import cx_Freeze  # 7.1.0


"""
$ ./build.py build      # 在当前目录新建包含可执行文件的子目录

或者

$ ./build.py bdist_msi  # 构建安装程序
"""

cx_Freeze.setup(
    name="PL-玛卡巴卡",
    version="0.1",
    options={
        "build_exe": {
            "build_exe": "bin",
            "packages": ["lark"],
            "optimize": 2,
            "include_msvcr": True,
            "no_compress": True,
            "zip_includes": ["pl.lark", "stdlib.txt"],
        }
    },
    executables=[cx_Freeze.Executable("pl.py")],
)
