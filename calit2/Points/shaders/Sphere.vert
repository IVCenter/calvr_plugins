#version 150 compatibility
#extension GL_ARB_gpu_shader5 : enable

void main(void)
{

    gl_FrontColor = gl_Color;

    // return projection position
    gl_Position = gl_ModelViewMatrix * gl_Vertex;
}
