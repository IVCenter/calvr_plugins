#version 120
#extension GL_EXT_gpu_shader4 : enable

/*
 * Fragment shader that stores the color value in first draw buffer texture
 */

varying out uvec4 fragcolor;

void main()
{
    fragcolor = uvec4(uint(gl_Color.r * 255),uint(gl_Color.g * 255),0,255);
}
