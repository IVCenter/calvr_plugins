#ifndef LIC_SHADERS_H
#define LIC_SHADERS_H

std::string licVertSrc =
"#version 150 compatibility                          \n"
"#extension GL_ARB_gpu_shader5 : enable              \n"
"                                                    \n"
"void main(void)                                     \n"
"{                                                   \n"
"    gl_Position = ftransform();                     \n"
"    gl_TexCoord[0] = gl_MultiTexCoord0;             \n"
"}                                                   \n";

std::string licFragSrc =
"#version 150 compatibility                          \n"
"                                                    \n"
"uniform sampler2D tex;                              \n"
"uniform float alpha;                                \n"
"                                                    \n"
"void main()                                         \n"
"{                                                   \n"
"    float color = texture2D(tex,gl_TexCoord[0].xy).r; \n"
"    gl_FragColor = vec4(color,color,color,alpha); \n"
"}                                                   \n";

#endif
