// 1x1 (pointwise) convolution, optional stride 2, then ReLU.
//
// NHWC vec4<f32> channel groups. Weights are flat f32 [out_c][in_c];
// biases are flat [out_c].

struct Params {
    a: vec4<u32>,  // n, ih, iw, oh
    b: vec4<u32>,  // ow, stride, in_vec4, out_vec4
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.a.x;
    let ih = params.a.y;
    let iw = params.a.z;
    let oh = params.a.w;
    let ow = params.b.x;
    let stride = params.b.y;
    let in_vec4 = params.b.z;
    let out_vec4 = params.b.w;
    let in_c = in_vec4 * 4u;

    let total = n * oh * ow * out_vec4;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let v4 = idx % out_vec4;
    let rem = idx / out_vec4;
    let ox = rem % ow;
    let rem2 = rem / ow;
    let oy = rem2 % oh;
    let n_i = rem2 / oh;

    var acc0 = bias[v4 * 4u + 0u];
    var acc1 = bias[v4 * 4u + 1u];
    var acc2 = bias[v4 * 4u + 2u];
    var acc3 = bias[v4 * 4u + 3u];

    let iy = oy * stride;
    let ix = ox * stride;
    let in_base = ((n_i * ih + iy) * iw + ix) * in_vec4;
    for (var ic = 0u; ic < in_vec4; ic++) {
        let in_v = input[in_base + ic];
        // Every output channel of this vec4 group consumes all four input
        // channels of the current input vec4 group.
        let wb = v4 * 4u * in_c + ic * 4u;
        acc0 += in_v.x * weights[wb + 0u]
              + in_v.y * weights[wb + 1u]
              + in_v.z * weights[wb + 2u]
              + in_v.w * weights[wb + 3u];
        acc1 += in_v.x * weights[wb + in_c + 0u]
              + in_v.y * weights[wb + in_c + 1u]
              + in_v.z * weights[wb + in_c + 2u]
              + in_v.w * weights[wb + in_c + 3u];
        acc2 += in_v.x * weights[wb + 2u * in_c + 0u]
              + in_v.y * weights[wb + 2u * in_c + 1u]
              + in_v.z * weights[wb + 2u * in_c + 2u]
              + in_v.w * weights[wb + 2u * in_c + 3u];
        acc3 += in_v.x * weights[wb + 3u * in_c + 0u]
              + in_v.y * weights[wb + 3u * in_c + 1u]
              + in_v.z * weights[wb + 3u * in_c + 2u]
              + in_v.w * weights[wb + 3u * in_c + 3u];
    }

    let obase = ((n_i * oh + oy) * ow + ox) * out_vec4 + v4;
    output[obase] = vec4<f32>(
        max(acc0, 0.0),
        max(acc1, 0.0),
        max(acc2, 0.0),
        max(acc3, 0.0),
    );
}
