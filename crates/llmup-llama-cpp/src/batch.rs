use std::ptr::null_mut;

use llmup_llama_cpp_sys::llama;

use crate::token::Token;

pub struct Batch {
    pub(crate) batch: llama::llama_batch,
    tokens_capacity: usize,
    embeddings_size: usize,
    nb_sequences_max: usize,
}

impl std::fmt::Debug for Batch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut display = String::new();
        display.push_str("[");
        for i in 0..self.batch.n_tokens as usize {
            if i > 0 {
                display.push_str(", ");
            }
            let tok = unsafe { self.batch.token.add(i).read() };
            let pos = unsafe { self.batch.pos.add(i).read() };
            let logit = unsafe { self.batch.logits.add(i).read() };
            display.push_str(&format!("{}-{}-{}", tok, pos, logit))
        }
        display.push_str("]");
        f.debug_struct("MyStruct")
            .field("allocated", &self.tokens_capacity)
            .field("embedding_size", &self.embeddings_size)
            .field("nb_sequences_max", &self.nb_sequences_max)
            .field("tokens", &display)
            .finish()
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        // workaround the mismatch between Drop API and the llama_batch_free API
        // as the free function free the content by taking the structure by value instead of pointer.
        //
        // copy all the pointers into the own arbitrary batch struct that will be freed with the API call
        //
        // other potential workaround will be to mark llama_batch Copy but bindgen doesn't have the ability
        // to do that selective for some types.
        let b = self.dup_batch();
        unsafe { llama::llama_batch_free(b) }
    }
}

impl Batch {
    /// effectively leak the llama batch structure, so that drop doesn't free
    /// all the memory fields when this is passed to llama_cpp functions
    pub(crate) fn dup_batch(&self) -> llama::llama_batch {
        llama::llama_batch {
            n_tokens: self.batch.n_tokens,
            token: self.batch.token,
            embd: self.batch.embd,
            pos: self.batch.pos,
            n_seq_id: self.batch.n_seq_id,
            seq_id: self.batch.seq_id,
            logits: self.batch.logits,
        }
    }

    pub fn new(tokens_capacity: usize, embeddings_size: usize, nb_sequences_max: usize) -> Self {
        let itokens_capacity = i32::try_from(tokens_capacity).expect("valid tokens usize");
        let iembeddings_size = i32::try_from(embeddings_size).expect("valid embedding usize");
        let inb_sequences_max =
            i32::try_from(nb_sequences_max).expect("valid nb_sequences_max usize");
        let batch = unsafe {
            llama::llama_batch_init(itokens_capacity, iembeddings_size, inb_sequences_max)
        };
        Self {
            batch,
            tokens_capacity,
            embeddings_size,
            nb_sequences_max,
        }
    }

    pub fn clear(&mut self) {
        if self.batch.n_tokens == 0 {
            return;
        }
        for i in 0..self.tokens_capacity {
            unsafe {
                self.batch.token.add(i).write(0);
                self.batch.n_seq_id.add(i).write(0);
                self.batch.logits.add(i).write(0);
            }
        }
        self.batch.n_tokens = 0;
    }

    pub fn from_tokens(tokens: &[Token], at: usize) -> Self {
        let mut batch = Batch::new(tokens.len(), 0, 1);
        for (i, token) in tokens.iter().enumerate() {
            let last = i == tokens.len() - 1;
            batch.append(*token, at + i, &[0], last);
        }
        //println!("from-token post filling: n-token {}", batch.batch.n_tokens);
        batch
    }

    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.tokens_capacity
    }

    pub fn is_embedding(&self) -> bool {
        self.batch.embd != null_mut()
    }

    pub fn append(&mut self, token: Token, position: usize, sequence_ids: &[i32], logits: bool) {
        let batch_idx = self.batch.n_tokens as usize;
        if batch_idx == self.tokens_capacity {
            panic!("cannot append");
        }

        if self.is_embedding() {
            panic!("cannot append token to embeddings")
        }

        if sequence_ids.len() > self.nb_sequences_max {
            panic!("too many element in sequences ids ")
        }

        let i_pos = i32::try_from(position).expect("valid position");

        let logit_i8 = if logits { 1 } else { 0 };
        unsafe {
            self.batch.token.add(batch_idx).write(token.0);
            self.batch.logits.add(batch_idx).write(logit_i8);
            self.batch.pos.add(batch_idx).write(i_pos);
            self.batch
                .n_seq_id
                .add(batch_idx)
                .write(sequence_ids.len() as i32);
            let seq_ptr = self.batch.seq_id.add(batch_idx).read();
            for (i, seqid) in sequence_ids.iter().enumerate() {
                seq_ptr.add(i).write(*seqid)
            }
        }

        self.batch.n_tokens += 1;
    }
}
