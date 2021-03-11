#!/bin/bash

isort liter/*.py
yapf -i liter/*.py

isort tests/*.py
yapf -i tests/*.py