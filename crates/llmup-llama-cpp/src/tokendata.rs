use llmup_llama_cpp_sys::llama;

use crate::Token;

#[repr(transparent)]
pub struct TokenData {
    data: llama::llama_token_data,
}

impl Clone for TokenData {
    fn clone(&self) -> Self {
        Self {
            data: llama::llama_token_data {
                id: self.data.id,
                logit: self.data.logit,
                p: self.data.p,
            },
        }
    }
}

impl TokenData {
    /// Create a new TokenData from a token, logit, and probability.
    pub fn new(token: Token, logit: f32, proba: f32) -> Self {
        TokenData {
            data: llama::llama_token_data {
                id: token.0,
                logit,
                p: proba,
            },
        }
    }

    pub const fn id(&self) -> Token {
        Token(self.data.id)
    }
    pub const fn set_id(&mut self, id: Token) {
        self.data.id = id.0
    }
    pub const fn logit(&self) -> f32 {
        self.data.logit
    }
    pub const fn set_logit(&mut self, logit: f32) {
        self.data.logit = logit
    }
    pub const fn proba(&self) -> f32 {
        self.data.p
    }
    pub const fn set_proba(&mut self, proba: f32) {
        self.data.p = proba
    }
}

pub struct TokenDataArray {
    pub data: Vec<TokenData>,
    pub selected: Option<usize>,
    pub sorted: bool,
}

impl TokenDataArray {
    pub fn as_mut_ptr<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(*mut llama::llama_token_data_array) -> R,
    {
        let mut c_struct = llama::llama_token_data_array {
            data: self.data.as_mut_ptr() as *mut llama::llama_token_data,
            size: self.data.len(),
            selected: self.selected.map(|u| u as i64).unwrap_or(-1),
            sorted: self.sorted,
        };

        let r = f(&mut c_struct);

        r
    }
}
