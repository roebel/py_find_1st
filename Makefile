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

sdist:
	$(PYTHON) setup.py sdist
	@echo now do
	@echo twine upload -r test dist/py_find_1st-x.y.z.tar.gz
	@echo for testing and
	@echo twine upload -r pypi dist/py_find_1st-x.y.z.tar.gz
	@echo for final distribution