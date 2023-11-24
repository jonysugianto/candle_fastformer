use candle_core::{CudaDevice, DType, Device, Module, Result, Tensor};
use candle_fastformer::fastformer::pos_embedding_sin::PosEmbeddingSin;
use candle_nn::{linear, VarBuilder, VarMap};
fn main() -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let max_seq_len: usize = 10;
    let embed_model_dim: usize = 8;
    let pos_embed = PosEmbeddingSin::new(&vb, max_seq_len, embed_model_dim)?;

    let xs = Tensor::rand(0.0, 1.0, &[2, 6, 8], &Device::Cpu)?.to_dtype(DType::F32)?;
    let out = pos_embed.forward(&xs)?;

    println!("out:{:?}", out);

    Ok(())
}
