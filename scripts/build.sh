#!/bin/bash

rm -r build
rm -r dist
rm -r torchliter.egg-info

python setup.py sdist bdist_wheel