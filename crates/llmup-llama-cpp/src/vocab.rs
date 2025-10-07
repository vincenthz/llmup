use std::{ffi::c_char, ptr::null_mut, sync::Arc};

use llmup_llama_cpp_sys::llama;

use crate::{Model, token::Token};

#[derive(Clone)]
#[allow(dead_code)]
pub struct Vocab {
    pub(crate) model: Model,
    pub(crate) ptr: Arc<VocabPtr>,
}

unsafe impl Send for Vocab {}
unsafe impl Sync for Vocab {}

pub struct VocabPtr(pub(crate) *const llama::llama_vocab);

impl Vocab {
    pub fn tokenize_size(&self, bytes: &[u8], first: bool) -> usize {
        let content_len = <i32>::try_from(bytes.len()).unwrap();
        let content = bytes.as_ptr() as *const c_char;

        unsafe {
            -llama::llama_tokenize(self.ptr.0, content, content_len, null_mut(), 0, first, true)
                as usize
        }
    }

    pub fn tokenize(&self, bytes: &[u8], first: bool) -> Vec<Token> {
        let content_len = <i32>::try_from(bytes.len()).unwrap();
        let content = bytes.as_ptr() as *const c_char;

        let size = self.tokenize_size(bytes, first);
        let mut out = Vec::with_capacity(size);

        let out_ptr = out.as_mut_ptr() as *mut llama::llama_token;

        let n = unsafe {
            llama::llama_tokenize(
                self.ptr.0,
                content,
                content_len,
                out_ptr,
                size as i32,
                first,
                true,
            )
        };
        //println!("{}", n);
        assert_eq!(n as usize, size);
        unsafe {
            out.set_len(size);
        }
        out
    }

    pub fn as_bytes(&self, token: Token) -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        let buf_ptr = buf.as_mut_ptr();
        let n = unsafe {
            llama::llama_token_to_piece(
                self.ptr.0,
                token.0,
                buf_ptr as *mut c_char,
                buf.len() as i32,
                0,
                true,
            )
        };

        if n < 0 {
            panic!("failed to convert token to piece")
        }
        buf.truncate(n as usize);
        buf
    }

    pub fn as_string_lossy(&self, token: Token) -> String {
        String::from_utf8_lossy(&self.as_bytes(token)).to_string()
    }

    pub fn as_string(&self, token: Token) -> String {
        String::from_utf8(self.as_bytes(token)).unwrap()
    }

    pub fn eos(&self) -> Token {
        Token(unsafe { llama::llama_vocab_eos(self.ptr.0) })
    }

    pub fn sep(&self) -> Token {
        Token(unsafe { llama::llama_vocab_sep(self.ptr.0) })
    }

    pub fn bos(&self) -> Token {
        Token(unsafe { llama::llama_vocab_bos(self.ptr.0) })
    }

    pub fn is_eog(&self, token: Token) -> bool {
        unsafe { llama::llama_vocab_is_eog(self.ptr.0, token.0) }
    }
}
