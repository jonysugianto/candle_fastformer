use candle_core::{CudaDevice, DType, Device, Module, Result, Tensor};
use candle_fastformer::fastformer::fastselfattention::FastSelfAttention;
use candle_nn::{linear, VarBuilder, VarMap};

fn main() -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

    let in_dim: usize = 20;
    let d_model: usize = 64;
    let num_heads: usize = 8;
    let model = FastSelfAttention::new(&vb, in_dim, d_model, num_heads, true)?;

    let q = Tensor::rand(0.0, 1.0, &[2, 4, 64], &Device::Cpu)?.to_dtype(DType::F32)?;
    let k = Tensor::rand(0.0, 1.0, &[2, 4, 64], &Device::Cpu)?.to_dtype(DType::F32)?;
    let v = Tensor::rand(0.0, 1.0, &[2, 4, 64], &Device::Cpu)?.to_dtype(DType::F32)?;

    let global_q = model.compute_global_query(&q)?;
    let global_k = model.compute_global_key(&global_q, &k)?;
    let r = model.compute_r(&global_k, &v)?;

    println!("global_q shape:{:?}", global_q.shape());
    println!("global_k:{:?}", global_k.shape());
    println!("r:{:?}", r.shape());

    let xs = Tensor::rand(0.0, 1.0, &[2, 4, in_dim], &Device::Cpu)?.to_dtype(DType::F32)?;
    let out = model.forward(&xs)?;
    println!("out:{:?}", out.shape());

    Ok(())
}
