use llmup_llama_cpp_sys::llama;

use crate::{Model, Sampler, Vocab, batch::Batch, token::Token};

#[allow(dead_code)]
pub struct Context {
    pub(crate) model: Model,
    pub(crate) tokens: usize,
    pub(crate) context_params: llama::llama_context_params,
    pub(crate) ptr: *mut llama::llama_context,
}

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

impl Context {
    pub fn new(model: Model, params: &ContextParams) -> Result<Self, ()> {
        let c_params = params.as_c();
        let c_params_clone = params.as_c();
        let ctx = unsafe { llama::llama_new_context_with_model(model.ptr.0, c_params) };
        if ctx.is_null() {
            return Err(());
        }

        Ok(Self {
            model,
            tokens: 0,
            context_params: c_params_clone,
            ptr: ctx,
        })
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

    pub fn advance(&mut self, vocab: &Vocab, sampler: &Sampler) {
        const DEBUG: bool = false;
        let mut batch = Batch::new(1, 0, 1);
        //let mut cleared = Vec::new();
        loop {
            let new_token = sampler.sample(self);

            if vocab.is_eog(new_token) {
                break;
            }

            let bytes = vocab.as_bytes(new_token);
            match str::from_utf8(&bytes) {
                Ok(s) => {
                    if DEBUG {
                        println!("token[{}] : {} ", self.tokens, s);
                    } else {
                        print!("{}", s);
                    }
                }
                Err(e) => {
                    if DEBUG {
                        println!("token[{}] : UTF8-Err {} : {:?}", self.tokens, e, bytes);
                    } else {
                    }
                }
            }
            use std::io::Write;
            std::io::stdout().flush().unwrap();

            batch.append(new_token, self.tokens, &[0], true);
            //println!("  batch {:?}", batch);

            loop {
                match self.decode(&batch) {
                    Ok(_) => {
                        break;
                    }
                    Err(DecodeError::CannotFindKVSlot) => {
                        //cleared.push(self.tokens);
                        //print!(" ------ ");
                        //self.memory_clear(false);
                    }
                    Err(e) => {
                        panic!("decode error {:?}", e)
                    }
                }
            }
            self.tokens += 1;
            if self.tokens >= 1024 {
                break;
            }

            batch.clear();
        }

        //println!("cleared: {:?}", cleared)
    }
}
