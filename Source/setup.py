from cx_Freeze import setup, Executable

setup(name = "ec" ,
      version = "0.1" ,
	  options = {"build_exe": {"packages": ["numpy"]}},
      description = "" ,
      executables = [Executable("ASL_Recognize.py")])