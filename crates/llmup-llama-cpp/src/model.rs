use llmup_llama_cpp_sys;

use llmup_llama_cpp_sys::llama;
use std::os::raw::c_char;
use std::path::Path;

use super::context::Context;

pub struct Model {
    ptr: *mut llama::llama_model,
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            llama::llama_free_model(self.ptr);
        }
    }
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ()> {
        let path = path.as_ref();
        let ret = unsafe {
            let params = llama::llama_model_default_params();
            llama::llama_load_model_from_file(path_to_cpath(path), params)
        };
        if ret.is_null() {
            return Err(());
        }

        Ok(Model { ptr: ret })
    }

    pub fn vocab(&self) -> Vocab {
        unsafe {
            Vocab {
                ptr: llama::llama_model_get_vocab(self.ptr),
            }
        }
    }

    pub fn new_context(&self) -> Result<Context, ()> {
        let ctx = unsafe {
            let context = llama::llama_context_default_params();
            llama::llama_new_context_with_model(self.ptr, context)
        };
        if ctx.is_null() {
            return Err(());
        }

        Ok(Context { ptr: ctx })
    }

    pub fn tokenize(&self, bytes: &[u8]) -> Result<Vec<Token>, ()> {
        let mut out = Vec::with_capacity(bytes.len() + 2);

        let content_len = <i32>::try_from(bytes.len()).unwrap();

        let content = bytes.as_ptr() as *const c_char;
        let out_ptr = out.as_mut_ptr() as *mut llama::llama_token;

        let n = unsafe {
            let vocab = llama::llama_model_get_vocab(self.ptr);
            llama::llama_tokenize(
                vocab,
                content,
                content_len,
                out_ptr,
                out.capacity() as i32,
                false,
                true,
            )
        };

        if n < 0 {
            return Err(());
        }

        unsafe {
            out.set_len(n as usize);
        }
        out.shrink_to_fit();
        Ok(out)
    }
}

#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;

use crate::token::Token;
use crate::vocab::Vocab;

#[cfg(unix)]
fn path_to_cpath(path: &Path) -> *const ::std::os::raw::c_char {
    path.as_os_str().as_bytes().as_ptr() as *const i8
}
