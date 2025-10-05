use llmup_llama_cpp_sys::llama;

use crate::{Model, Sampler, Vocab, batch::Batch, token::Token};

#[allow(dead_code)]
pub struct Context {
    pub(crate) model: Model,
    pub(crate) tokens: usize,
    pub(crate) context_params: llama::llama_context_params,
    pub(crate) ptr: *mut llama::llama_context,
}

unsafe impl Send for Context {}

pub struct ContextParams {
    n_ctx: u32,
}

impl Default for ContextParams {
    fn default() -> Self {
        let mut context = unsafe { llama::llama_context_default_params() };
        context.n_ctx = 16384;
        Self {
            n_ctx: context.n_ctx,
        }
    }
}

impl ContextParams {
    fn as_c(&self) -> llama::llama_context_params {
        let mut context = unsafe { llama::llama_context_default_params() };
        context.n_ctx = self.n_ctx;
        context
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DecodeError {
    CannotFindKVSlot,
    Aborted,
    InvalidBatch,
    UnspecifiedWarning(#[allow(dead_code)] i32),
    FatalError(#[allow(dead_code)] i32),
}

#[derive(Debug)]
pub struct ContextCreateError;

impl std::fmt::Display for ContextCreateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Context Create Error (consult logging for reasons)")
    }
}

impl std::error::Error for ContextCreateError {}

impl Context {
    pub fn new(model: Model, params: &ContextParams) -> Result<Self, ContextCreateError> {
        let c_params = params.as_c();
        let c_params_clone = params.as_c();
        let ctx = unsafe { llama::llama_new_context_with_model(model.ptr.0, c_params) };
        if ctx.is_null() {
            return Err(ContextCreateError);
        }

        Ok(Self {
            model,
            tokens: 0,
            context_params: c_params_clone,
            ptr: ctx,
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn state_get(&self) -> Vec<u8> {
        let state_size = unsafe { llama::llama_state_get_size(self.ptr) };

        let mut state = vec![0; state_size];
        unsafe {
            llama::llama_state_get_data(self.ptr, state.as_mut_ptr(), state_size);
        }
        state
    }

    pub fn state_set(&mut self, data: &[u8]) {
        let read = unsafe { llama::llama_state_set_data(self.ptr, data.as_ptr(), data.len()) };
        assert_eq!(read, data.len());
    }

    pub fn memory_clear(&self, clear_data: bool) {
        unsafe {
            let memory = llama::llama_get_memory(self.ptr);
            llama::llama_memory_clear(memory, clear_data)
        }
    }

    fn decode(&self, batch: &Batch) -> Result<(), DecodeError> {
        let b = batch.dup_batch();
        let ret = unsafe { llama::llama_decode(self.ptr, b) };
        match ret {
            0 => Ok(()),
            1 => Err(DecodeError::CannotFindKVSlot),
            2 => Err(DecodeError::Aborted),
            -1 => Err(DecodeError::InvalidBatch),
            _ if ret > 0 => Err(DecodeError::UnspecifiedWarning(ret)),
            _ => Err(DecodeError::FatalError(ret)),
        }
    }

    pub fn append_tokens(&mut self, tokens: &[Token]) {
        let batch = Batch::from_tokens(tokens, self.tokens);
        self.tokens += tokens.len();
        self.decode(&batch).unwrap();
    }

    pub fn next_token(&mut self, sampler: &Sampler, vocab: &Vocab) -> Option<Token> {
        let new_token = sampler.sample(self);
        (!vocab.is_eog(new_token)).then_some(new_token)
    }
}
