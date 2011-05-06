#ifndef GL_HELPER_H
#define GL_HELPER_H

#include <GL/glew.h>
#include <GL/gl.h>
#include <string>

bool checkShader(GLuint shader);
bool checkProgram(GLuint program);
bool checkFramebuffer();
bool checkGL();
bool createShader(std::string file, GLenum shaderType, GLuint & shader);
bool createProgram(GLuint & program, GLuint vertexShader, GLuint fragShader, GLuint geometryShader = 0, GLenum inputType = GL_TRIANGLES, GLenum outputType = GL_TRIANGLE_STRIP, int vertOut = 3); 


#endif
