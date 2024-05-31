#! python3.12
import cx_Freeze  # 7.1.0


"""
$ ./build.py build
"""

cx_Freeze.setup(
    name="plm",
    version="0.1.0",
    options={
        "build_exe": {
            "packages": ["lark"],
        }
    },
    executables=[cx_Freeze.Executable("pl.py")],
)
