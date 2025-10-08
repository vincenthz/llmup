use llmup_llama_cpp_sys::llama;
use thiserror::Error;

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
    pub n_ctx: u32,
    pub embeddings: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        let mut context = unsafe { llama::llama_context_default_params() };
        context.n_ctx = 16384;
        Self {
            n_ctx: context.n_ctx,
            embeddings: context.embeddings,
        }
    }
}

impl ContextParams {
    fn as_c(&self) -> llama::llama_context_params {
        let mut context = unsafe { llama::llama_context_default_params() };
        context.n_ctx = self.n_ctx;
        context.embeddings = self.embeddings;
        context
    }
}

#[derive(Clone, Copy, Debug, Error)]
pub enum DecodeError {
    #[error("cannot find KV Slot")]
    CannotFindKVSlot,
    #[error("Aborted")]
    Aborted,
    #[error("Invalid Batch")]
    InvalidBatch,
    #[error("Unspecified Decode Warning {0}")]
    UnspecifiedWarning(#[allow(dead_code)] i32),
    #[error("Fatal Decode Error {0}")]
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

#[derive(Debug, Clone)]
pub struct EmbeddingSeqError(i32);

impl std::fmt::Display for EmbeddingSeqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Embedding Sequence Error {}", self.0)
    }
}

impl std::error::Error for EmbeddingSeqError {}

#[derive(Debug, Error)]
pub enum ContextEmbeddingError {
    #[error("embedding decoding failed: {0}")]
    DecodeError(#[from] DecodeError),
    #[error("embedding sequence getting failed: {0}")]
    EmbeddingSeqError(#[from] EmbeddingSeqError),
    #[error("pooling type not yet supported: {0:?}")]
    UnsupportedPoolingType(PoolingType),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingType {
    Unspecified,
    None,
    Mean,
    Cls,
    Last,
    Rank,
}

impl From<llama::llama_pooling_type> for PoolingType {
    fn from(value: llama::llama_pooling_type) -> Self {
        match value {
            llama::llama_pooling_type::LLAMA_POOLING_TYPE_UNSPECIFIED => Self::Unspecified,
            llama::llama_pooling_type::LLAMA_POOLING_TYPE_NONE => Self::None,
            llama::llama_pooling_type::LLAMA_POOLING_TYPE_MEAN => Self::Mean,
            llama::llama_pooling_type::LLAMA_POOLING_TYPE_CLS => Self::Cls,
            llama::llama_pooling_type::LLAMA_POOLING_TYPE_LAST => Self::Last,
            llama::llama_pooling_type::LLAMA_POOLING_TYPE_RANK => Self::Rank,
            _ => Self::Unspecified,
        }
    }
}

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

    pub fn embeddings(&mut self, tokens: &[Token]) -> Result<Vec<f32>, ContextEmbeddingError> {
        let pooling_type = self.pooling_type();
        if pooling_type != PoolingType::Mean {
            return Err(ContextEmbeddingError::UnsupportedPoolingType(pooling_type));
        }

        self.append_tokens(&tokens)?;

        let e = self.embeddings_seq_ith(0)?;
        let s = e.iter().map(|f| f * f).sum::<f32>().sqrt();
        let norm = if s > 0.0 { 1.0 / s } else { 0.0 };
        let mut e = e.to_vec();

        for f in e.iter_mut() {
            *f *= norm;
        }
        Ok(e)
    }

    pub fn n_ctx(&self) -> u32 {
        unsafe { llama::llama_n_ctx(self.ptr) }
    }

    pub fn n_batch(&self) -> u32 {
        unsafe { llama::llama_n_batch(self.ptr) }
    }

    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama::llama_n_ubatch(self.ptr) }
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

    pub fn append_tokens(&mut self, tokens: &[Token]) -> Result<(), DecodeError> {
        let batch = Batch::from_tokens(tokens, self.tokens);
        self.decode(&batch)?;
        self.tokens += tokens.len();
        Ok(())
    }

    pub fn next_token<S: Sampler>(&mut self, sampler: &mut S, vocab: &Vocab) -> Option<Token> {
        let new_token = sampler.sample(self, -1);
        (!vocab.is_eog(new_token)).then_some(new_token)
    }

    pub fn pooling_type(&self) -> PoolingType {
        let pooling_type = unsafe { llama::llama_pooling_type(self.ptr) };
        pooling_type.into()
    }

    pub fn embeddings_seq_ith(&self, i: i32) -> Result<&[f32], EmbeddingSeqError> {
        let n_embd = self.model.n_embd() as usize;

        unsafe {
            let embedding = llama::llama_get_embeddings_seq(self.ptr, i);
            if embedding == core::ptr::null_mut() {
                return Err(EmbeddingSeqError(i));
            }

            Ok(core::slice::from_raw_parts(embedding, n_embd))
        }
    }
}
