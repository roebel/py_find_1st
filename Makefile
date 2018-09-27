all: build

PYTHON=python

build : Makefile utils_find_1st/find_1st.cpp
	$(PYTHON) setup.py build_ext 

cythonize : 

install:
	pip install .

install-user:
	pip install --user .

clean:
	$(PYTHON) setup.py clean -a
