package_dir = utils_find_1st
package_prefix=py_find_1st

PYTHON=python

PYPROJECT = pyproject.toml
VERSION = $(shell egrep '^ *version_str *=' $(package_dir)/__init__.py | cut -d'"' -f2)

DIST_DIR = dist
PLATFORM_TAG = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_platform().replace('-', '_').replace('.', '_'))")
PYTHON_VERSION = $(shell $(PYTHON) -c "import sys; print('cp{}{}'.format(sys.version_info[0], sys.version_info[1]))")
WHEEL_FILE = $(DIST_DIR)/$(package_prefix)-$(VERSION)-$(PYTHON_VERSION)-$(PYTHON_VERSION)-$(PLATFORM_TAG).whl
SDIST_FILE = $(DIST_DIR)/$(package_prefix)-$(VERSION).tar.gz
NUMPYVERSION=$(shell $(PYTHON) -c "from importlib.metadata import version; print(version('numpy'));" )

.PHONY: build sdist


all: build

PYTHON=python


build : $(WHEEL_FILE) 
sdist: $(SDIST_FILE)
	@echo now do
	@echo twine upload -r test dist/py_find_1st-$(VERSION).tar.gz
	@echo for testing and
	@echo twine upload -r pypi dist/py_find_1st-$(VERSION).tar.gz
	@echo for final distribution
	@echo in case you want to try a clean install from test.pypi.org use
	@echo pip install --no-cache-dir --extra-index-url https://test.pypi.org/simple/  py_find_1st==${VERSION} 

install:
	pip install .

install-user:
	pip install --user .


$(WHEEL_FILE): Makefile README.md utils_find_1st/__init__.py utils_find_1st/find_1st.cpp test/test_find_1st.py setup.py pyproject.toml 
	$(PYTHON) -m build . --wheel


clean:
	rm -rf build $(WHEEL_FILE) $(SDIST_FILE)
	rm -rf test/utf1st_inst_dir/*


$(SDIST_FILE): Makefile README.md utils_find_1st/__init__.py setup.py pyproject.toml utils_find_1st/find_1st.cpp test/test_find_1st.py 
	$(PYTHON) -m build . --sdist

check: build
	pip install $(WHEEL_FILE) --find-links=dist --no-deps  --upgrade --target test/utf1st_inst_dir
	touch test/utf1st_inst_dir/__init__.py
	pwd; cd ./test; $(PYTHON) test_find_1st.py
