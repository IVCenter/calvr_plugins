#version 120
#extension GL_EXT_gpu_shader4 : enable
#extension GL_ARB_texture_rectangle : enable

uniform usampler2DRect colors[8];
uniform sampler2DRect depth[8];
uniform sampler2DRect depthR8[4];
uniform int textures;
uniform float redLookup[32];
uniform float greenLookup[32];
uniform float blueLookup[32];
uniform float ncolors;

void main()
{
    // read color from texture
    uvec4 color = texture2DRect(colors[0],gl_FragCoord.xy);

    // read depth from texture
    float dpth = texture2DRect(depth[0],gl_FragCoord.xy).r;

    // do color lookup
    int colorIndex = int((float(color.r) / 255.0 * ncolors) + 0.15);
    vec3 colorLookup = vec3(redLookup[colorIndex],greenLookup[colorIndex],blueLookup[colorIndex]);

    // multiply by lighting
    gl_FragColor.rgb = colorLookup.rgb * (float(color.g) / 255.0);
    gl_FragDepth = dpth;
}
