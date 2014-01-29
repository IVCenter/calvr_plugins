#ifndef LIC_SHADERS_H
#define LIC_SHADERS_H

std::string licVertSrc =
"#version 150 compatibility                          \n"
"#extension GL_ARB_gpu_shader5 : enable              \n"
"                                                    \n"
"void main(void)                                     \n"
"{                                                   \n"
"    gl_Position = ftransform();                     \n"
"}                                                   \n";

std::string licFragSrc =
"#version 150 compatibility                          \n"
"                                                    \n"
"uniform sampler2D tex;                              \n"
"                                                    \n"
"void main()                                         \n"
"{                                                   \n"
"    float color = texture2D(tex,gl_TexCoord[0].xy).r; \n"
"    gl_FragColor = vec4(color,color,color,1.0); \n"
"}                                                   \n";

#endif
