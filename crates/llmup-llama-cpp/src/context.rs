use llmup_llama_cpp_sys::llama;

use crate::{Model, Sampler, Vocab, batch::Batch, token::Token};

#[allow(dead_code)]
pub struct Context {
    pub(crate) model: Model,
    pub(crate) tokens: usize,
    pub(crate) ptr: *mut llama::llama_context,
}

#[derive(Clone, Copy, Debug)]
pub enum DecodeError {
    CannotFindKVSlot,
    Aborted,
    InvalidBatch,
    UnspecifiedWarning(i32),
    FatalError(i32),
}

impl Context {
    pub fn new(model: Model) -> Result<Self, ()> {
        let ctx = unsafe {
            let context = llama::llama_context_default_params();
            llama::llama_new_context_with_model(model.ptr.0, context)
        };
        if ctx.is_null() {
            return Err(());
        }

        Ok(Self {
            model,
            tokens: 0,
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
                        self.memory_clear(false);
                    }
                    Err(e) => {
                        panic!("decode error {:?}", e)
                    }
                }
            }
            self.tokens += 1;

            batch.clear();
        }

        //println!("cleared: {:?}", cleared)
    }
}
