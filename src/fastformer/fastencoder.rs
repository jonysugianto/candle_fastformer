use super::fastencoder_block::FastEncoderBlock;
use candle_core::Module;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::linear_no_bias;
use candle_nn::Linear;
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct FastEncoder<PE: Module> {
    pub blocks: Vec<FastEncoderBlock>,
    pub input_embedding: Linear,
    pub pos_embed: PE,
}

impl<PE: Module> FastEncoder<PE> {
    pub fn new(
        vb: &VarBuilder,
        d_model: usize,
        d_input: usize,
        num_heads: usize,
        ff_d_hidden: usize,
        num_blocks: usize,
        pe: PE,
    ) -> Result<Self> {
        let mut blocks: Vec<FastEncoderBlock> = Vec::new();
        for _ in 0..num_blocks {
            let b = FastEncoderBlock::new(vb, d_model, num_heads, ff_d_hidden)?;
            blocks.push(b);
        }

        let input_embedding = linear_no_bias(d_input, d_model, vb.pp("input_embedding"))?;
        let ret = FastEncoder {
            blocks: blocks,
            pos_embed: pe,
            input_embedding: input_embedding,
        };

        return Ok(ret);
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let pxs = self.input_embedding.forward(xs)?;
        let mut out = self.pos_embed.forward(&pxs)?;
        for l in &self.blocks {
            out = l.forward(&out)?;
        }
        return Ok(out);
    }
}
