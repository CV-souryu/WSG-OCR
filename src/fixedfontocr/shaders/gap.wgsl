// Global average pooling over H x W.
//
// NHWC vec4<f32> channel groups. One invocation computes one vec4 (four
// channels) for one glyph by summing every spatial position.

struct Params {
    a: vec4<u32>,  // n, ih, iw, vec4_count
    b: vec4<u32>,  // unused
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.a.x;
    let ih = params.a.y;
    let iw = params.a.z;
    let vec4_count = params.a.w;

    let total = n * vec4_count;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let v4 = idx % vec4_count;
    let n_i = idx / vec4_count;

    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var y = 0u; y < ih; y++) {
        for (var x = 0u; x < iw; x++) {
            acc += input[((n_i * ih + y) * iw + x) * vec4_count + v4];
        }
    }
    output[idx] = acc / f32(ih * iw);
}
