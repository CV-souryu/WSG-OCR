// Phase-1 GPU entry point.
//
// Input:  uint8 glyph batch [N, H, W] packed as u32 words
//         (byte 0 of a word is the first pixel, little-endian).
// Output: NHWC f32 glyph batch [N, H, W, 4] as vec4<f32> elements.
//         Only channel 0 is meaningful; the other three are zero padding
//         so every downstream layer works on vec4<f32> channel groups.

struct Params {
    a: vec4<u32>,  // n, h, w, unused
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.a.x * params.a.y * params.a.z;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let word = input[idx / 4u];
    let byte = (word >> ((idx % 4u) * 8u)) & 0xffu;
    let v = f32(byte) * (1.0 / 255.0);
    output[idx] = vec4<f32>(v, 0.0, 0.0, 0.0);
}
