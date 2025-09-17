use llmup_llama_cpp_sys::llama;

use crate::{Sampler, Vocab, token::Token};

pub struct Context {
    pub(crate) ptr: *mut llama::llama_context,
}

impl Drop for Context {
    fn drop(&mut self) {}
}

impl Context {
    pub fn advance(&mut self, tokens: &mut [Token], vocab: &Vocab, sampler: &Sampler) {
        let tokens_ptr = tokens.as_mut_ptr();
        let mut batch =
            unsafe { llama::llama_batch_get_one(tokens_ptr as *mut i32, tokens.len() as i32) };

        let mut new_token_id: llama::llama_token;
        loop {
            let ret = unsafe { llama::llama_decode(self.ptr, batch) };
            if ret != 0 {
                panic!("decode failed");
            }

            let new_token = sampler.sample(self);
            new_token_id = new_token.0;

            if vocab.is_eog(new_token) {
                break;
            }

            let s = vocab.as_string(Token(new_token_id));
            print!("{}", s);
            use std::io::Write;
            std::io::stdout().flush().unwrap();

            batch = unsafe { llama::llama_batch_get_one(&mut new_token_id, 1) }
        }
    }
}
