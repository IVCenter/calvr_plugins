#version 120
#extension GL_EXT_gpu_shader4 : enable
#extension GL_ARB_texture_rectangle : enable

uniform usampler2DRect colors[8];
uniform usampler2DRect depth[8];
uniform usampler2DRect depthR8[4];
uniform int textures;
uniform float redLookup[32];
uniform float greenLookup[32];
uniform float blueLookup[32];
uniform float ncolors;

void main()
{
    // read values from gpu 0
    uvec4 color = texture2DRect(colors[0],gl_FragCoord.xy);
    float dpth = 1.0 - (float((texture2DRect(depth[0],gl_FragCoord.xy).r << uint(8)) + texture2DRect(depthR8[0],gl_FragCoord.xy).r) / 16777215.0);

    // read depth from gpu 1
    float temp = 1.0 - (float((texture2DRect(depth[1],gl_FragCoord.xy).r << uint(8)) + texture2DRect(depthR8[1],gl_FragCoord.xy).r) / 16777215.0);

    // if less, replace values
    if(temp < dpth)
    {
	dpth = temp;
	color = texture2DRect(colors[1],gl_FragCoord.xy);
    }

    // read depth from gpu 2
    temp = 1.0 - (float((texture2DRect(depth[2],gl_FragCoord.xy).r << uint(8)) + texture2DRect(depthR8[2],gl_FragCoord.xy).r) / 16777215.0);

    // if less, replace values
    if(temp < dpth)
    {
	dpth = temp;
	color = texture2DRect(colors[2],gl_FragCoord.xy);
    }

    // read depth from gpu 3
    temp = 1.0 - (float((texture2DRect(depth[3],gl_FragCoord.xy).r << uint(8)) + texture2DRect(depthR8[3],gl_FragCoord.xy).r) / 16777215.0);

    // if less, replace values
    if(temp < dpth)
    {
	dpth = temp;
	color = texture2DRect(colors[3],gl_FragCoord.xy);
    }

    // perform color lookup
    int colorIndex = int((float(color.r) / 255.0 * ncolors) + 0.15);
    vec3 colorLookup = vec3(redLookup[colorIndex],greenLookup[colorIndex],blueLookup[colorIndex]);

    // multiply by lighting
    gl_FragColor.rgb = colorLookup.rgb * (float(color.g) / 255.0);
    gl_FragDepth = dpth;
}
