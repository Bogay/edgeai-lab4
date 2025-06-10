bin := "./build/my_whisper_app"

gen:
    mkdir -p build
    cd build && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_BLAS=1 ..

alias b := build
build: gen
    cmake --build build --config Release

bench: build
    #!/usr/bin/env bash
    for m in $(ls ./models/*.bin); do
        v=$(basename "$m")
        {{ bin }} -l zh \
            $m audio references.txt > output_$v.txt
    done
    
    
