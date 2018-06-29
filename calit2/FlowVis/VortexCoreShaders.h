#ifndef VORTEX_CORE_SHADERS_H
#define VORTEX_CORE_SHADERS_H

std::string vcoreAlphaVertSrc =
"#version 150 compatibility \n"
"#extension GL_ARB_gpu_shader5 : enable \n"
"#extension GL_ARB_explicit_attrib_location : enable \n"
"\n"
"layout(location = 4) in float value; \n"
"\n"
"uniform float min; \n"
"uniform float max; \n"
"\n"
"void main(void) \n"
"{ \n"
"    gl_Position = ftransform(); \n"
"    gl_FrontColor = gl_Color; \n"
"    gl_FrontColor.a = clamp((0.6 * (value-min) / (max-min) + 0.4),0.0,1.0); \n"
"} \n";

#endif
