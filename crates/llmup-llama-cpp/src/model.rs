use llmup_llama_cpp_sys;

use llmup_llama_cpp_sys::llama;
use std::path::Path;
use std::ptr::null_mut;
use std::sync::Arc;

use crate::Vocab;
use crate::context::ContextParams;
use crate::vocab::VocabPtr;

use super::context::{Context, ContextCreateError};

#[derive(Clone)]
pub struct Model {
    pub(crate) ptr: Arc<ModelPtr>,
}

pub struct ModelPtr(pub(crate) *mut llama::llama_model);

impl Drop for ModelPtr {
    fn drop(&mut self) {
        unsafe {
            llama::llama_free_model(self.0);
        }
    }
}

#[derive(Default)]
pub struct ModelParams {
    pub vocab_only: bool,
}

impl ModelParams {
    fn as_c(&self) -> llama::llama_model_params {
        let mut params = unsafe { llama::llama_model_default_params() };

        params.vocab_only = self.vocab_only;
        params
    }
}

#[derive(Debug)]
pub struct ModelLoadError;

impl std::fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Model Load Error (consult logging for reasons)")
    }
}

impl std::error::Error for ModelLoadError {}

impl Model {
    pub fn load(path: impl AsRef<Path>, params: &ModelParams) -> Result<Self, ModelLoadError> {
        let path = path.as_ref();
        let c_params = params.as_c();
        let ret = unsafe { llama::llama_load_model_from_file(path_to_cpath(path), c_params) };
        if ret.is_null() {
            return Err(ModelLoadError);
        }

        Ok(Model {
            ptr: Arc::new(ModelPtr(ret)),
        })
    }

    pub fn vocab(&self) -> Vocab {
        unsafe {
            Vocab {
                model: self.clone(),
                ptr: Arc::new(VocabPtr(llama::llama_model_get_vocab(self.ptr.0))),
            }
        }
    }

    /// Get the model type
    pub fn description(&self) -> String {
        let sz = unsafe { llama::llama_model_desc(self.ptr.0, null_mut(), 0) as usize };
        let mut buf = vec![0; sz];
        unsafe { llama::llama_model_desc(self.ptr.0, buf.as_mut_ptr() as *mut i8, sz) };
        String::from_utf8(buf).unwrap()
    }

    pub fn has_encoder(&self) -> bool {
        unsafe { llama::llama_model_has_encoder(self.ptr.0) }
    }

    pub fn has_decoder(&self) -> bool {
        unsafe { llama::llama_model_has_decoder(self.ptr.0) }
    }

    /// Create a new context for this model
    pub fn new_context(&self, params: &ContextParams) -> Result<Context, ContextCreateError> {
        Context::new(self.clone(), params)
    }
}

#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;

#[cfg(unix)]
fn path_to_cpath(path: &Path) -> *const ::std::os::raw::c_char {
    path.as_os_str().as_bytes().as_ptr() as *const i8
}
