use std::{ffi::c_char, ptr::null_mut, sync::Arc};

use llmup_llama_cpp_sys::llama::{self, llama_token_attr};

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

#[derive(Clone, Copy, Debug)]
pub enum VocabType {
    SPM,
    BPE,
    WPM,
    UGM,
    RWKV,
    PLAMO2,
}

#[derive(Clone, Copy, Debug)]
pub struct TokenAttr(u32);

impl TokenAttr {
    pub fn is_undefined(self) -> bool {
        self.0 == llama_token_attr::LLAMA_TOKEN_ATTR_UNDEFINED as u32
    }

    pub fn is_unknown(self) -> bool {
        (self.0 & llama_token_attr::LLAMA_TOKEN_ATTR_UNKNOWN as u32) != 0
    }

    pub fn is_unused(self) -> bool {
        (self.0 & llama_token_attr::LLAMA_TOKEN_ATTR_UNUSED as u32) != 0
    }

    pub fn is_normal(self) -> bool {
        (self.0 & llama_token_attr::LLAMA_TOKEN_ATTR_NORMAL as u32) != 0
    }

    pub fn is_control(self) -> bool {
        (self.0 & llama_token_attr::LLAMA_TOKEN_ATTR_CONTROL as u32) != 0
    }

    pub fn is_user_defined(self) -> bool {
        (self.0 & llama_token_attr::LLAMA_TOKEN_ATTR_USER_DEFINED as u32) != 0
    }

    pub fn is_byte(self) -> bool {
        (self.0 & llama_token_attr::LLAMA_TOKEN_ATTR_BYTE as u32) != 0
    }

    pub fn is_normalized(self) -> bool {
        (self.0 & llama_token_attr::LLAMA_TOKEN_ATTR_NORMALIZED as u32) != 0
    }

    pub fn is_lstrip(self) -> bool {
        (self.0 & llama_token_attr::LLAMA_TOKEN_ATTR_LSTRIP as u32) != 0
    }

    pub fn is_rstrip(self) -> bool {
        (self.0 & llama_token_attr::LLAMA_TOKEN_ATTR_RSTRIP as u32) != 0
    }

    pub fn is_single_word(self) -> bool {
        (self.0 & llama_token_attr::LLAMA_TOKEN_ATTR_SINGLE_WORD as u32) != 0
    }
}

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

    pub fn n_tokens(&self) -> u32 {
        unsafe { llama::llama_vocab_n_tokens(self.ptr.0) as u32 }
    }

    pub fn tokens(&self) -> impl Iterator<Item = Token> {
        (0..self.n_tokens()).map(|i| Token(i as i32))
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

    pub fn token_attr(&self, token: Token) -> TokenAttr {
        unsafe { TokenAttr(llama::llama_token_get_attr(self.ptr.0, token.0) as u32) }
    }

    pub fn vocab_type(&self) -> Option<VocabType> {
        unsafe {
            let ty = llama::llama_vocab_type(self.ptr.0);
            match ty {
                llama::llama_vocab_type::LLAMA_VOCAB_TYPE_NONE => None,
                llama::llama_vocab_type::LLAMA_VOCAB_TYPE_SPM => Some(VocabType::SPM),
                llama::llama_vocab_type::LLAMA_VOCAB_TYPE_BPE => Some(VocabType::BPE),
                llama::llama_vocab_type::LLAMA_VOCAB_TYPE_WPM => Some(VocabType::WPM),
                llama::llama_vocab_type::LLAMA_VOCAB_TYPE_UGM => Some(VocabType::UGM),
                llama::llama_vocab_type::LLAMA_VOCAB_TYPE_RWKV => Some(VocabType::RWKV),
                llama::llama_vocab_type::LLAMA_VOCAB_TYPE_PLAMO2 => Some(VocabType::PLAMO2),
                _ => None,
            }
        }
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
