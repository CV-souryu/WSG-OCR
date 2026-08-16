// preprocess_soft.wgsl — GPU soft glyph batch (Goal 20 G3).
//
// One dispatch turns an uploaded RGB image + per-glyph ROI/frame params into
// the packed uint8 [N, 24, 24] soft glyph batch the mega shader consumes.
// Each thread computes one u32 word = 4 output pixels:
//
//   value(oy, ox) = gray(image[sy, sx])   if inside the placed frame
//                   0                     otherwise
//
// where sy = y + trunc(i*h/new_h), sx = x + trunc(j*w/new_w) — the exact
// nearest-neighbor resample of normalize_grayscale/_resample (f32 truncation
// is bit-identical to the numpy f64 computation because the fractional parts
// are >= 1/24 and the products stay below 2^24). gray = r*0.299 + g*0.587 +
// b*0.114 in f32, matching the default profile's soft_foreground formula.
// The host precomputes new_h/new_w/y0/x0 (f64 round-half-even placement).
//
// Output layout: n*144 u32 words, the same packed-byte layout mega.wgsl
// reads as input, so both dispatches can share one command encoder.

struct Params {
    a: vec4<u32>,  // n, H, W, unused
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> img_words: array<u32>;
@group(0) @binding(2) var<storage, read> glyph_params: array<u32>;  // n*8
@group(0) @binding(3) var<storage, read_write> output: array<u32>;  // n*144

fn byte_at(offset: u32) -> u32 {
    let word = img_words[offset / 4u];
    return (word >> ((offset % 4u) * 8u)) & 0xffu;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.a.x;
    let H = params.a.y;
    let W = params.a.z;
    let total = n * 144u;
    let wid = gid.x;
    if (wid >= total) {
        return;
    }
    let g = wid / 144u;
    let k = wid % 144u;
    let base = g * 8u;
    let x = glyph_params[base + 0u];
    let y = glyph_params[base + 1u];
    let w = glyph_params[base + 2u];
    let h = glyph_params[base + 3u];
    let new_h = glyph_params[base + 4u];
    let new_w = glyph_params[base + 5u];
    let y0 = glyph_params[base + 6u];
    let x0 = glyph_params[base + 7u];

    var word = 0u;
    for (var j = 0u; j < 4u; j++) {
        let byte_pos = k * 4u + j;
        let oy = byte_pos / 24u;
        let ox = byte_pos % 24u;
        var v = 0u;
        if (new_h != 0u && new_w != 0u
            && oy >= y0 && oy < y0 + new_h
            && ox >= x0 && ox < x0 + new_w) {
            let i = oy - y0;
            let jj = ox - x0;
            let sy = y + u32(f32(i * h) / f32(new_h));
            let sx = x + u32(f32(jj * w) / f32(new_w));
            let rgb = (sy * W + sx) * 3u;
            var f = 0.0;
            f += f32(byte_at(rgb + 0u)) * 0.299;
            f += f32(byte_at(rgb + 1u)) * 0.587;
            f += f32(byte_at(rgb + 2u)) * 0.114;
            v = u32(f);
        }
        word |= v << (j * 8u);
    }
    output[wid] = word;
}
