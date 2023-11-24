use candle_core::{DType, Device, Result, Tensor};
use candle_fastformer::fastformer::{fastencoder::FastEncoder, pos_embedding_sin::PosEmbeddingSin};
use candle_nn::{VarBuilder, VarMap};
fn main() -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let d_model: usize = 64;
    let d_input: usize = 20;
    let ff_d_hidden = 128;
    let num_heads: usize = 8;
    let num_blocks = 3;

    let max_seq_len: usize = 10;
    let pos_embed = PosEmbeddingSin::new(&vb, max_seq_len, d_model)?;

    let model = FastEncoder::new(
        &vb,
        d_model,
        d_input,
        num_heads,
        ff_d_hidden,
        num_blocks,
        pos_embed,
    )?;

    let xs = Tensor::rand(0.0, 1.0, &[2, 4, d_input], &Device::Cpu)?.to_dtype(DType::F32)?;
    let out = model.forward(&xs)?;

    println!("out shape:{:?}", out.shape());
    println!("out:{:?}", out);

    Ok(())
}
