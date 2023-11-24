use candle_core::{CudaDevice, DType, Device, Module, Result, Tensor};
use candle_fastformer::fastformer::fastencoder_block::FastEncoderBlock;
use candle_nn::{linear, VarBuilder, VarMap};
fn main() -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let dim: usize = 64;
    let ff_d_hidden = 128;
    let num_heads: usize = 8;
    let model = FastEncoderBlock::new(&vb, dim, num_heads, ff_d_hidden)?;

    let xs = Tensor::rand(0.0, 1.0, &[2, 4, dim], &Device::Cpu)?.to_dtype(DType::F32)?;
    let out = model.forward(&xs)?;

    println!("out shape:{:?}", out.shape());
    println!("out:{:?}", out);

    Ok(())
}
