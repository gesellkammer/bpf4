#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y atlas-devel

# Compile wheels
# for PYBIN in /opt/python/*/bin; do
 for PYBIN in /opt/python/cp36*/bin; do
    "${PYBIN}/pip" install -r /io/requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# pip install -r /io/requirements.txt
# pip wheel /io/ -w wheelhouse/

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/cp36*/bin/; do
     "${PYBIN}/pip" install bpf4 --no-index -f /io/wheelhouse
     (cd "$HOME"; "${PYBIN}/nosetests" bpf4)
done
