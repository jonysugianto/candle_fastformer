use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use rand_distr::num_traits::ToPrimitive;

#[derive(Debug)]
pub struct PosEmbeddingSin {
    pub max_seq_len: usize,
    pub embed_model_dim: usize,
    pub pe: Tensor,
}

impl PosEmbeddingSin {
    pub fn new(
        vb: &VarBuilder,
        max_seq_len: usize,
        embed_model_dim: usize,
    ) -> Result<PosEmbeddingSin> {
        let mut values: Vec<Vec<f32>> = Vec::new();

        for pos in 0..max_seq_len {
            let mut vs: Vec<f32> = Vec::new();
            for i in (0..embed_model_dim).step_by(2) {
                let ipw = ((2 * i) / embed_model_dim).to_f64().unwrap();
                let ipw = (10000.0f64).powf(ipw);
                let ipw = pos.to_f64().unwrap() / ipw;
                let v1 = ipw.sin();

                let i1pw = ((2 * (i + 1)) / embed_model_dim).to_f64().unwrap();
                let i1pw = (10000.0f64).powf(i1pw);
                let i1pw = pos.to_f64().unwrap() / i1pw;
                let v2 = i1pw.sin();

                vs.push(v1.to_f32().unwrap());
                vs.push(v2.to_f32().unwrap());
            }
            values.push(vs);
        }

        let pe = Tensor::new(values, &candle_core::Device::Cpu)?;
        Ok(Self {
            max_seq_len: max_seq_len,
            embed_model_dim: embed_model_dim,
            pe: pe,
        })
    }
}

impl Module for PosEmbeddingSin {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        let seq_len = dims[1];
        let pe = self.pe.i((..seq_len, ..))?;
        let pe = pe.unsqueeze(0)?.repeat((dims[0]))?;
        let out = (xs + &pe);
        out
    }
}
