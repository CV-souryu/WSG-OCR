// First-layer 3x3 convolution, SAME zero padding, stride 2, then ReLU.
//
// The TinyCNN's only standard convolution is conv1 (1 -> 8 input channels).
// The input is stored as one vec4<f32> per pixel with the real channel in
// component x (components y/z/w are zero padding), so each output channel
// accumulates input.x only. Weights are flat f32 [out_c][3][3]; biases are
// flat [out_c].

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
    let _in_vec4 = params.b.z;
    let out_vec4 = params.b.w;

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

    for (var dy = 0u; dy < 3u; dy++) {
        let iy = i32(oy * stride) + i32(dy) - 1;
        if (iy < 0 || iy >= i32(ih)) {
            continue;
        }
        for (var dx = 0u; dx < 3u; dx++) {
            let ix = i32(ox * stride) + i32(dx) - 1;
            if (ix < 0 || ix >= i32(iw)) {
                continue;
            }
            let in_v = input[(n_i * ih + u32(iy)) * iw + u32(ix)];
            let wb = v4 * 36u + dy * 3u + dx;
            acc0 += in_v.x * weights[wb + 0u * 9u];
            acc1 += in_v.x * weights[wb + 1u * 9u];
            acc2 += in_v.x * weights[wb + 2u * 9u];
            acc3 += in_v.x * weights[wb + 3u * 9u];
        }
    }

    let obase = ((n_i * oh + oy) * ow + ox) * out_vec4 + v4;
    output[obase] = vec4<f32>(
        max(acc0, 0.0),
        max(acc1, 0.0),
        max(acc2, 0.0),
        max(acc3, 0.0),
    );
}
