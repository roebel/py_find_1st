all: build

PYTHON=python
vv=$(shell $(PYTHON) setup.py get_version )
.PHONY: build

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
	@echo twine upload -r test dist/py_find_1st-$(vv).tar.gz
	@echo for testing and
	@echo twine upload -r pypi dist/py_find_1st-$(vv).tar.gz
	@echo for final distribution
	@echo in case you want to try a clean install from test.pypi.org use
	@echo pip install --no-cache-dir --extra-index-url https://test.pypi.org/simple/  py_find_1st==${vv} 

check: build
	pip install --no-build-isolation --no-deps --no-cache --upgrade --target test/utf1st_inst_dir .
	touch test/utf1st_inst_dir/__init__.py
	pwd; cd ./test; $(PYTHON) test_find_1st.py
