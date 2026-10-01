// The walls of a folder's cube, seen from the inside.
//
// With BACKDROP (an open folder), every wall fragment sits at the far plane:
// it shows only where nothing else in the scene was drawn, so walls never
// hide what lies behind or inside them.

#import bevy_pbr::forward_io::VertexOutput

@group(2) @binding(0) var<uniform> color: vec4<f32>;

struct WallOutput {
    @location(0) color: vec4<f32>,
#ifdef BACKDROP
    @builtin(frag_depth) depth: f32,
#endif
}

@fragment
fn fragment(in: VertexOutput) -> WallOutput {
    // A fixed shade per face, so the cube reads as a box without lighting.
    let facing = abs(normalize(in.world_normal));
    let shade = 0.8 + 0.2 * dot(facing, vec3<f32>(0.3, 0.55, 0.15));
    var out: WallOutput;
    out.color = vec4<f32>(color.rgb * shade, color.a);
#ifdef BACKDROP
    // Reverse Z: zero is the far plane.
    out.depth = 0.0;
#endif
    return out;
}
