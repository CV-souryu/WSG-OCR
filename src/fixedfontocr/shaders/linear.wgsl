// Dense layer: [N, in_c] -> [N, out_c].
// Weights are flat f32 [out_c][in_c]; biases are flat [out_c].

struct Params {
    a: vec4<u32>,  // n, in_c, out_c, unused
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.a.x;
    let in_c = params.a.y;
    let out_c = params.a.z;

    let total = n * out_c;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let oc = idx % out_c;
    let n_i = idx / out_c;
    var acc = bias[oc];
    for (var ic = 0u; ic < in_c; ic++) {
        acc += input[n_i * in_c + ic] * weights[oc * in_c + ic];
    }
    output[idx] = acc;
}
