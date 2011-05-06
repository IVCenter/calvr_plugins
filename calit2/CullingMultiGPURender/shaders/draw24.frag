#version 120
#extension GL_EXT_gpu_shader4 : enable

/*
 * Fragment shader that stores the color value in first draw buffer texture
 * and the depth value across a 16bit and an 8bit texture
 */

varying out uvec4 fragcolor;
varying out uvec4 fragdepthl;
varying out uvec4 fragdepths;

void main()
{
    fragcolor = uvec4(uint(gl_Color.r * 255),uint(gl_Color.g * 255),0,255);

    // (1.0 - x) is needed because the depth is stored in a color texture
    // that clears to zero instead of one, this is reversed in the
    // combination shader
    float depth = 1.0 - gl_FragCoord.z;

    // divide the depth range into 16bit and 8bit parts
    unsigned int depthi = uint(float(16777215) * depth);
    unsigned int depthsmall = depthi & uint(0xFF);

    // store depth parts in the textures
    fragdepths.r = depthsmall;
    depthi = depthi - depthsmall;
    fragdepthl.r = depthi >> uint(8);
}
