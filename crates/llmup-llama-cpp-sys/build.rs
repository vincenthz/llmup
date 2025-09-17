use std::{
    env,
    path::{Path, PathBuf},
};

use bindgen::Builder;
use cc::Build;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Cpu,
}

const WARNING_FLAGS: &[&str] = &[
    //"-Wshadow",
    "-Wstrict-prototypes",
    "-Wpointer-arith",
    "-Wmissing-prototypes",
    "-Werror=implicit-int",
    "-Werror=implicit-function-declaration",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Wcast-qual",
    "-Wno-unused-function",
    "-Wunreachable-code-break",
    "-Wunreachable-code-return",
    //"-Wdouble-promotion",
];

fn main() {
    let lib_path = std::path::PathBuf::from("libs");

    if !lib_path.exists() {
        panic!("cannot compile without cloning the llama.cpp git submodule")
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings(&lib_path, &out_path);
    let ggml_objects = lib_ggml(&lib_path);
    lib_llama(&lib_path, ggml_objects);
}

fn bindings(lib_path: &Path, out_path: &Path) {
    let ggml_path = lib_path.join("ggml");
    let ggml_include_path = ggml_path.join("include");
    let llama_include_path = lib_path.join("include");

    let ggml_bindings = Builder::default()
        .header(ggml_include_path.join("ggml.h").to_string_lossy())
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .derive_debug(true)
        .derive_copy(false)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .generate()
        .expect("Unable to generate bindings");

    let llama_bindings = Builder::default()
        .header(llama_include_path.join("llama.h").to_string_lossy())
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .clang_arg(format!("-I{}", ggml_include_path.display()))
        .derive_debug(true)
        .derive_copy(false)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .generate()
        .expect("Unable to generate bindings");

    ggml_bindings
        .write_to_file(out_path.join("bindings_ggml.rs"))
        .expect("cannot write ggml bindings");
    llama_bindings
        .write_to_file(out_path.join("bindings_llama.rs"))
        .expect("cannot write llama bindings")
}

fn lib_ggml(lib_path: &Path) -> Vec<PathBuf> {
    let ggml_path = lib_path.join("ggml");
    let src_path = ggml_path.join("src");
    let include_path = ggml_path.join("include");

    let c_files = ["ggml.c", "ggml-alloc.c", "ggml-quants.c"];

    let cpp_files = [
        "ggml.cpp",
        "ggml-backend.cpp",
        "ggml-opt.cpp",
        "ggml-threading.cpp",
        "gguf.cpp",
    ];

    // create a common build
    let mut common = Build::new();
    common.include(include_path);
    common.include(&src_path);
    common.opt_level(3);
    common.flags(WARNING_FLAGS);
    common.flags(["-MD", "-MT"]);

    common.define("GGML_SCHED_MAX_COPIES", "4");
    common.define("GGML_SHARED", None);
    common.define("GGML_BUILD", None);
    common.define("GGML_VERSION", "\"0.0.6458\"");
    common.define("GGML_COMMIT", "\"40be5115\"");
    common.define("ggml_base_EXPORTS", None);

    common.define("GGML_USE_CPU", None);

    // specialize for CPP
    let mut cpp = common.clone();

    cpp.cpp(true).std("c++17");
    cpp.files(cpp_files.into_iter().map(|f| src_path.join(f)));

    // specialize for C
    let mut c = common;
    c.cpp(false).std("c11");
    c.files(c_files.into_iter().map(|f| src_path.join(f)));

    let backend = Backend::Cpu;
    match backend {
        Backend::Cpu => {
            let backend_dir = src_path.join("ggml-cpu");
            c.include(&backend_dir);
            cpp.include(&backend_dir);
            let c_files = ["ggml-cpu.c", "quants.c", "arch/arm/quants.c"];
            let cpp_files = [
                "ggml-cpu.cpp",
                "repack.cpp",
                "hbm.cpp",
                "traits.cpp",
                "amx/amx.cpp",
                "amx/mmq.cpp",
                "binary-ops.cpp",
                "unary-ops.cpp",
                "vec.cpp",
                "ops.cpp",
                "llamafile/sgemm.cpp",
                "arch/arm/repack.cpp",
            ];

            c.files(c_files.into_iter().map(|f| backend_dir.join(f)));
            cpp.files(cpp_files.into_iter().map(|f| backend_dir.join(f)));
        }
    }

    cpp.file(src_path.join("ggml-backend-reg.cpp"));

    let res_cpp = cpp.compile_intermediates();
    let res_c = c.compile_intermediates();

    let mut all = vec![];
    all.extend(res_cpp.into_iter());
    all.extend(res_c.into_iter());
    all
}

fn lib_llama(lib_path: &Path, ggml_objects: Vec<PathBuf>) {
    let ggml_path = lib_path.join("ggml");
    let ggml_include_path = ggml_path.join("include");
    let src_path = lib_path.join("src");
    let include_path = lib_path.join("include");

    let cpp_files = [
        "llama.cpp",
        "llama-adapter.cpp",
        "llama-arch.cpp",
        "llama-batch.cpp",
        "llama-chat.cpp",
        "llama-context.cpp",
        "llama-cparams.cpp",
        "llama-grammar.cpp",
        "llama-graph.cpp",
        "llama-hparams.cpp",
        "llama-impl.cpp",
        "llama-io.cpp",
        "llama-kv-cache.cpp",
        "llama-kv-cache-iswa.cpp",
        "llama-memory.cpp",
        "llama-memory-hybrid.cpp",
        "llama-memory-recurrent.cpp",
        "llama-mmap.cpp",
        "llama-model-loader.cpp",
        "llama-model-saver.cpp",
        "llama-model.cpp",
        "llama-quant.cpp",
        "llama-sampling.cpp",
        "llama-vocab.cpp",
        "unicode-data.cpp",
        "unicode.cpp",
    ];

    // create a common build
    let mut common = Build::new();
    common.include(include_path);
    common.include(ggml_include_path);
    common.include(&src_path);
    common.opt_level(3);
    common.flags(WARNING_FLAGS);
    common.flags(["-MD", "-MT"]);
    common.define("GGML_SCHED_MAX_COPIES", "4");
    common.define("GGML_SHARED", None);
    common.define("GGML_BUILD", None);
    common.define("GGML_VERSION", "\"0.0.6458\"");
    common.define("GGML_COMMIT", "\"40be5115\"");
    common.define("ggml_base_EXPORTS", None);

    common.define("GGML_USE_CPU", None);

    // specialize for CPP
    let mut cpp = common;

    cpp.cpp(true).std("c++17");
    cpp.objects(ggml_objects);
    cpp.files(cpp_files.into_iter().map(|f| src_path.join(f)));

    cpp.compile("llama");
}
