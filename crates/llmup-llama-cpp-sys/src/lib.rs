#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

extern crate link_cplusplus;

pub mod ggml {
    include!(concat!(env!("OUT_DIR"), "/bindings_ggml.rs"));
}
pub mod llama {
    include!(concat!(env!("OUT_DIR"), "/bindings_llama.rs"));
}
