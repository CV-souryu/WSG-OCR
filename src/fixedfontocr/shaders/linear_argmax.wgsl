// Fused dense + argmax (plan: Linear and Argmax are merged).
//
// One 64-thread workgroup per glyph. Each thread computes the logits for a
// disjoint subset of classes directly from the [N, in_c] features, keeps its
// local top-1/top-2, and thread 0 merges the per-thread winners.
//
// Output: one 12-byte record per glyph:
//   best_id (u32), best_score (f32), second_score (f32).
// Confidence is the margin best_score - second_score, computed by the host.

struct Params {
    a: vec4<u32>,  // n, in_c, out_c, unused
};

struct Result {
    best_id: u32,
    best_score: f32,
    second_score: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> results: array<Result>;

var<workgroup> shared_score: array<f32, 64>;
var<workgroup> shared_id: array<u32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = wgid.x;
    let tid = lid.x;
    let in_c = params.a.y;
    let out_c = params.a.z;

    var s1 = -3.402823466e+38;
    var i1 = 0u;
    var s2 = -3.402823466e+38;
    var i2 = 0u;
    for (var c = tid; c < out_c; c += 64u) {
        var acc = bias[c];
        for (var ic = 0u; ic < in_c; ic++) {
            acc += features[n * in_c + ic] * weights[c * in_c + ic];
        }
        if (acc > s1) {
            s2 = s1;
            i2 = i1;
            s1 = acc;
            i1 = c;
        } else if (acc > s2) {
            s2 = acc;
            i2 = c;
        }
    }

    shared_score[tid] = s1;
    shared_id[tid] = i1;
    workgroupBarrier();

    if (tid == 0u) {
        var r1 = shared_score[0u];
        var r1_id = shared_id[0u];
        var r2 = -3.402823466e+38;
        var r2_id = 0u;
        for (var i = 1u; i < 64u; i++) {
            let s = shared_score[i];
            let id = shared_id[i];
            if (s > r1) {
                r2 = r1;
                r2_id = r1_id;
                r1 = s;
                r1_id = id;
            } else if (s > r2) {
                r2 = s;
                r2_id = id;
            }
        }
        results[n] = Result(r1_id, r1, r2);
    }
}
