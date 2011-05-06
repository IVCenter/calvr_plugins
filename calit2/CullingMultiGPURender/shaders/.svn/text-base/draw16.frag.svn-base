#version 120
#extension GL_EXT_gpu_shader4 : enable

/*
 * Fragment shader that stores the color value in first draw buffer texture
 * and the depth value in a second 16 bit texture
 */

varying out uvec4 fragcolor;
varying out uvec4 fragdepthl;

void main()
{
    fragcolor = uvec4(uint(gl_Color.r * 255),uint(gl_Color.g * 255),0,255);

    // (1.0 - x) is needed because the depth is stored in a color texture
    // that clears to zero instead of one, this is reversed in the
    // combination shader
    float dpth = 1.0 - gl_FragCoord.z;

    // find 16bit unsigned int value
    fragdepthl.r = uint(float(65535) * dpth);
}
