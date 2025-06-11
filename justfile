bin := "./build/my_whisper_app"
gen_flags := ("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    )
build_dir := "build"

gen:
    cmake {{ gen_flags }} -B {{ build_dir }}

alias b := build
build: gen
    cmake --build {{ build_dir }} --config Release

bench: build
    #!/usr/bin/env bash
    for m in $(ls ./models/*.bin); do
        v=$(basename "$m")
        {{ bin }} -l zh \
            $m audio references.txt > output_$v.txt
    done

clean:
    rm -rf {{ build_dir }}
