use llmup_llama_cpp_sys;

use llmup_llama_cpp_sys::llama;
use std::path::Path;

use crate::Vocab;

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
}

#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;

#[cfg(unix)]
fn path_to_cpath(path: &Path) -> *const ::std::os::raw::c_char {
    path.as_os_str().as_bytes().as_ptr() as *const i8
}
