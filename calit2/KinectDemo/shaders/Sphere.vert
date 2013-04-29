#version 150 compatibility
#extension GL_ARB_gpu_shader5 : enable
attribute vec3 morphvertex;
void main(void)
{

    gl_FrontColor = gl_Color;

    // return projection position
    gl_Position = gl_ModelViewMatrix * vec4(morphvertex,1.0);
}
