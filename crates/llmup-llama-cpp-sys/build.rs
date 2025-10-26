use std::{
    env,
    io::Write,
    path::{Path, PathBuf},
};

use bindgen::Builder;
use cc::Build;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(unused)]
enum Backend {
    Metal,
}

#[cfg(target_os = "macos")]
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

#[cfg(not(target_os = "macos"))]
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
    //"-Wdouble-promotion",
];

fn main() {
    let lib_path = std::path::PathBuf::from("libs");

    if !lib_path.exists() {
        panic!("cannot compile without cloning the llama.cpp git submodule")
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings(&lib_path, &out_path);
    let ggml_objects = lib_ggml(&lib_path, &out_path);
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

fn lib_ggml(lib_path: &Path, out_path: &Path) -> Vec<PathBuf> {
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

    common.define("GGML_SCHED_MAX_COPIES", "4");
    common.define("GGML_SHARED", None);
    common.define("GGML_BUILD", None);
    common.define("GGML_VERSION", "\"0.0.6458\"");
    common.define("GGML_COMMIT", "\"40be5115\"");
    common.define("ggml_base_EXPORTS", None);

    common.define("GGML_USE_CPU", None);

    let metaldefs = [
        "GGML_USE_METAL",
        "GGML_METAL_EMBED_LIBRARY",
        "ggml_metal_EXPORTS",
    ];

    // specialize for CPP
    let mut cpp = common.clone();

    cpp.cpp(true).std("c++17");
    cpp.files(cpp_files.into_iter().map(|f| src_path.join(f)));

    // specialize for C
    let mut c = common.clone();
    c.cpp(false).std("c11");
    c.files(c_files.into_iter().map(|f| src_path.join(f)));

    let mut backend_intermediates = Vec::new();

    // support cpu
    {
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

    #[cfg(target_os = "macos")]
    let backend = Some(Backend::Metal);
    #[cfg(not(target_os = "macos"))]
    let backend = None;

    match backend {
        Some(Backend::Metal) => {
            for metaldef in metaldefs {
                c.define(metaldef, None);
                cpp.define(metaldef, None);
            }

            let backend_dir = src_path.join("ggml-metal");
            c.include(&backend_dir);
            cpp.include(&backend_dir);

            let c_files = ["ggml-metal-context.m", "ggml-metal-device.m"];
            let cpp_files = [
                "ggml-metal.cpp",
                "ggml-metal-device.cpp",
                "ggml-metal-common.cpp",
                "ggml-metal-ops.cpp",
            ];

            cpp.define("GGML_METAL_EMBED_LIBRARY", None);

            let ggml_common_h_path = src_path.join("ggml-common.h");
            let ggml_common_h_data = std::fs::read_to_string(ggml_common_h_path).unwrap();

            let ggml_metal_impl_h_path = backend_dir.join("ggml-metal-impl.h");
            let ggml_metal_impl_h_data = std::fs::read_to_string(ggml_metal_impl_h_path).unwrap();

            let embed_file_name = "ggml-metal-embed.s";

            let metal_file = "ggml-metal.metal";
            let metal_file_path = backend_dir.join(metal_file);
            let metal_file_out = out_path.join(metal_file);

            let metal_file_data = std::fs::read_to_string(metal_file_path).unwrap();

            let replace_embed_ggml = "__embed_ggml-common.h__\n";
            let replace_ggml_metal = "#include \"ggml-metal-impl.h\"\n";

            let metal_file_data = metal_file_data
                .replace(replace_embed_ggml, &ggml_common_h_data)
                .replace(replace_ggml_metal, &ggml_metal_impl_h_data);

            let mut out_metal_file = std::fs::File::create(&metal_file_out).unwrap();
            out_metal_file
                .write_all(metal_file_data.as_bytes())
                .unwrap();

            let prefix = r#".section __DATA,__ggml_metallib
.globl _ggml_metallib_start
_ggml_metallib_start:
"#;
            let suffix = r#".globl _ggml_metallib_end
_ggml_metallib_end:
"#;

            let embed_file_path = out_path.join(embed_file_name);
            let mut out_embed_file = std::fs::File::create(&embed_file_path).unwrap();

            let embed_path = format!(".incbin \"{}\"\n", metal_file_out.display());
            out_embed_file.write_all(prefix.as_bytes()).unwrap();
            out_embed_file.write_all(embed_path.as_bytes()).unwrap();
            out_embed_file.write_all(suffix.as_bytes()).unwrap();

            let mut s = common.clone();
            s.cpp(false).std("c11");

            s.file(embed_file_path);
            let res_s = s.compile_intermediates();

            backend_intermediates.extend(res_s);

            c.files(c_files.into_iter().map(|f| backend_dir.join(f)));
            cpp.files(cpp_files.into_iter().map(|f| backend_dir.join(f)));

            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
            println!("cargo:rustc-link-lib=framework=MetalKit");

            //
            println!("cargo:rustc-cfg=feature_metal");
        }
        None => {}
    }

    cpp.file(src_path.join("ggml-backend-reg.cpp"));

    let res_cpp = cpp.compile_intermediates();
    let res_c = c.compile_intermediates();

    let mut all = vec![];
    all.extend(res_cpp.into_iter());
    all.extend(res_c.into_iter());
    all.extend(backend_intermediates.into_iter());
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
