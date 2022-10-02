#!/bin/bash

lazydocs --output-path "docs" \
    --src-base-url "https://github.com/ChenchaoZhao/TorchLiter/tree/main" \
    --overview-file "index.md" \
    --no-remove-package-prefix \
    --no-watermark \
    --no-validate \
    src/

