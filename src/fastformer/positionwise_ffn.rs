use candle_core::Module;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::linear_no_bias;
use candle_nn::Linear;
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct PositionWiseFFN {
    pub dense1: Linear,
    pub dense2: Linear,
}

impl PositionWiseFFN {
    pub fn new(vb: &VarBuilder, d_model: usize, d_hidden: usize) -> Result<PositionWiseFFN> {
        let dense1 = linear_no_bias(d_model, d_hidden, vb.pp("dense1"))?;
        let dense2 = linear_no_bias(d_hidden, d_model, vb.pp("dense2"))?;

        Ok(Self {
            dense1: dense1,
            dense2: dense2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.dense1.forward(x)?;
        let h2 = h.relu()?;
        let out = self.dense2.forward(&h2);
        return out;
    }
}
