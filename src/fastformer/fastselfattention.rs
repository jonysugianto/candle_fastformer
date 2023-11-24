use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{linear, Linear, VarBuilder};

#[derive(Debug)]
pub struct FastSelfAttention {
    pub to_qkv: Linear,
    pub to_alpha: Linear,
    pub to_beta: Linear,
    pub to_r: Linear,
    pub to_out: Linear,
    pub num_heads: usize,
    pub d_model: usize,
    pub scale: f64,
}

impl FastSelfAttention {
    pub fn new(
        vb: &VarBuilder,
        in_dim: usize,
        d_model: usize,
        num_heads: usize,
        qkv_bias: bool,
    ) -> Result<Self> {
        let to_qkv = if qkv_bias {
            linear(in_dim, d_model * 3, vb.pp("to_qkv"))
        } else {
            candle_nn::linear_no_bias(in_dim, d_model * 3, vb.pp("to_qkv"))
        }?;

        let to_alpha = linear(d_model, 1, vb.pp("to_alpha"))?;
        let to_beta = linear(d_model, 1, vb.pp("to_beta"))?;
        let to_r = linear(d_model, d_model, vb.pp("to_r"))?;
        let to_out = linear(d_model, d_model, vb.pp("to_out"))?;

        let scale = 1. / ((d_model / num_heads) as f64).sqrt();
        Ok(Self {
            d_model: d_model,
            to_qkv: to_qkv,
            to_alpha: to_alpha,
            to_beta: to_beta,
            to_r: to_r,
            to_out: to_out,
            num_heads: num_heads,
            scale: scale,
        })
    }

    pub fn compute_global_query(&self, q: &Tensor) -> Result<Tensor> {
        let alpha = self.to_alpha.forward(q)?;
        let lastdim = alpha.dims().len() - 1;
        let alpha = alpha.squeeze(lastdim)?;
        let alpha = softmax_last_dim(&alpha)?;
        let alpha = alpha.unsqueeze(lastdim - 1)?;
        let global_q = alpha.matmul(q)?;
        let global_q = global_q.squeeze(lastdim - 1);
        return global_q;
    }

    pub fn compute_global_key(&self, global_q: &Tensor, k: &Tensor) -> Result<Tensor> {
        let global_q = global_q.unsqueeze(1)?;
        let global_q = global_q.repeat((1, k.dims()[1], 1))?;
        let k_global_q = (global_q * k)?;
        let beta = self.to_beta.forward(&k_global_q)?;
        let lastdim = beta.dims().len() - 1;
        let beta = beta.squeeze(lastdim)?;
        let beta = softmax_last_dim(&beta)?;
        let beta = beta.unsqueeze(lastdim - 1)?;
        let global_key = beta.matmul(&k_global_q)?;
        let global_key = global_key.squeeze(lastdim - 1);
        return global_key;
    }

    pub fn compute_r(&self, global_k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let global_k = global_k.unsqueeze(1)?;
        let global_k = global_k.repeat((1, v.dims()[1], 1))?;
        let u = (global_k * v)?;
        let r = self.to_r.forward(&u);
        return r;
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self.to_qkv.forward(xs)?;
        let qkv = qkv.reshape((b, n, 3, self.num_heads, self.d_model / self.num_heads))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;

        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        let q = q.permute((0, 2, 1, 3))?;
        let dims = q.dims();
        let q = q.reshape((dims[0], dims[1], dims[2] * dims[3]))?;

        let k = k.permute((0, 2, 1, 3))?;
        let dims = k.dims();
        let k = k.reshape((dims[0], dims[1], dims[2] * dims[3]))?;

        let v = v.permute((0, 2, 1, 3))?;
        let dims = v.dims();
        let v = v.reshape((dims[0], dims[1], dims[2] * dims[3]))?;

        let global_q = self.compute_global_query(&q)?;
        let global_k = self.compute_global_key(&global_q, &k)?;
        let r = self.compute_r(&global_k, &v)?;
        let r = (&r + &q)?;
        let out = self.to_out.forward(&r);
        return out;
    }
}
