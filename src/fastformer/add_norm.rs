use candle_core::Module;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::LayerNorm;
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct AddNorm {
    pub norm: LayerNorm,
}

impl AddNorm {
    pub fn new(vb: &VarBuilder, d_model: usize) -> Result<AddNorm> {
        let wn = Tensor::new(&[1.0f32], &candle_core::Device::Cpu)?;
        let n = LayerNorm::new_no_bias(wn, 1e-5);

        Ok(Self { norm: n })
    }

    pub fn forward(&self, x: &Tensor, fx: &Tensor) -> Result<Tensor> {
        let out = (x + fx)?;
        let out = self.norm.forward(&out);
        return out;
    }
}
