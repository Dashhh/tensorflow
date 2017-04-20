BUILD TENSORFLOWA:
1. w tensorflow_source/tensorflow trzeba uruchomic ./configure, dalej podac sciezke do pythona3.5 (/usr/bin/python3.5) i reszte domyslnie
2. uruchomic ./build_pip.sh, zeby zbudowac i zaimportowac wheel-a do pythona

UZYCIE BIBLIOTEKI
3. w tensorflow_source/tensorflow/tensorflow/core/usr_ops uruchomic skrypt build_script.sh, zeby zbudowac biblioteke z conv_bin
4. przykladowy kod pythonowy, ktory tego uzywa, jest w tensorflow_source (example_script.py)

DEBUG
5. zeby puscic sobie kod pythonowy z tensorflowa z wlaczonym valgrindem, jest skrypt dla example_script.py -> run_python_valgrind.sh
