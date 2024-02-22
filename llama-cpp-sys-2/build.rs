use std::env;
use std::path::Path;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=llama.cpp");

    let cublas_enabled = env::var("CARGO_FEATURE_CUBLAS").is_ok();

    if !Path::new("llama.cpp/ggml.c").exists() {
        panic!("llama.cpp seems to not be populated, try running `git submodule update --init --recursive` to init.")
    }

    let mut ggml = cc::Build::new();
    let mut ggml_cuda = if cublas_enabled {
        Some(cc::Build::new())
    } else {
        None
    };
    let mut llama_cpp = cc::Build::new();

    ggml.cpp(false);
    llama_cpp.cpp(true);

    // https://github.com/ggerganov/llama.cpp/blob/a836c8f534ab789b02da149fbdaf7735500bff74/Makefile#L364-L368
    if let Some(ggml_cuda) = &mut ggml_cuda {
        for lib in ["cuda", "cublas", "cudart", "cublasLt"] {
            println!("cargo:rustc-link-lib={}", lib);
        }
        if !ggml_cuda.get_compiler().is_like_msvc() {
            for lib in ["culibos", "pthread", "dl", "rt"] {
                println!("cargo:rustc-link-lib={}", lib);
            }
        }

        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

        if cfg!(target_arch = "aarch64") {
            ggml_cuda
                .flag_if_supported("-mfp16-format=ieee")
                .flag_if_supported("-mno-unaligned-access");
            llama_cpp
                .flag_if_supported("-mfp16-format=ieee")
                .flag_if_supported("-mno-unaligned-access");
            ggml_cuda
                .flag_if_supported("-mfp16-format=ieee")
                .flag_if_supported("-mno-unaligned-access");
        }

        ggml_cuda
            .cuda(true)
            .flag("-arch=all")
            .file("llama.cpp/ggml-cuda.cu");

        if ggml_cuda.get_compiler().is_like_msvc() {
            ggml_cuda.std("c++14");
        } else {
            ggml_cuda.std("c++17");
        }

        ggml.define("GGML_USE_CUBLAS", None);
        ggml_cuda.define("GGML_USE_CUBLAS", None);
        llama_cpp.define("GGML_USE_CUBLAS", None);
    }

    ggml.flag_if_supported("-march=native")
        .flag_if_supported("-mtune=native");
    llama_cpp
        .flag_if_supported("-march=native")
        .flag_if_supported("-mtune=native");
    if let Some(ggml_cuda) = &mut ggml_cuda {
        ggml_cuda
            .flag_if_supported("-march=native")
            .flag_if_supported("-mtune=native");
    }

    // https://github.com/ggerganov/llama.cpp/blob/191221178f51b6e81122c5bda0fd79620e547d07/Makefile#L133-L141
    if cfg!(target_os = "macos") {
        llama_cpp.define("_DARWIN_C_SOURCE", None);
    }
    if cfg!(target_os = "dragonfly") {
        llama_cpp.define("__BSD_VISIBLE", None);
    }

    if let Some(ggml_cuda) = ggml_cuda {
        println!("compiling ggml-cuda");
        ggml_cuda.compile("ggml-cuda");
    }

    if cfg!(target_os = "linux") {
        ggml.define("_GNU_SOURCE", None);
    }

    ggml.std("c17")
        .file("llama.cpp/ggml.c")
        .file("llama.cpp/ggml-alloc.c")
        .file("llama.cpp/ggml-backend.c")
        .file("llama.cpp/ggml-quants.c")
        .define("GGML_USE_K_QUANTS", None);

    llama_cpp
        .define("_XOPEN_SOURCE", Some("600"))
        .std("c++17")
        .file("llama.cpp/llama.cpp");

    println!("compiling ggml");
    ggml.compile("ggml");

    println!("compiling llama");
    llama_cpp.compile("llama");

    let header = "llama.cpp/llama.h";

    println!("cargo:rerun-if-changed={header}");

    let bindings = bindgen::builder()
        .header(header)
        .derive_partialeq(true)
        .no_debug("llama_grammar_element")
        .prepend_enum_name(false)
        .derive_eq(true)
        .generate()
        .expect("failed to generate bindings for llama.cpp");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write bindings to file");
}
