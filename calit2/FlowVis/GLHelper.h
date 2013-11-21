/**
 * @file GLHelper.h
 * Contains functions to perform some opengl operations 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#ifndef GL_HELPER_H
#define GL_HELPER_H

#include <GL/glew.h>
#include <GL/gl.h>
#include <string>

/// checks for errors creating an opengl shader
bool checkShader(GLuint shader, std::string name = "");

/// checks for errors linking an opengl shader program
bool checkProgram(GLuint program);

/// checks for errors in a bound frame buffer object
bool checkFramebuffer();

/// checks for general opengl errors
bool checkGL();

/// creates a shader program from a file name
bool createShader(std::string file, GLenum shaderType, GLuint & shader);

bool createShaderFromSrc(std::string & source, GLenum shaderType, GLuint & shader, std::string name = "");

/// creates a shader program from a number of shaders
bool createProgram(GLuint & program, GLuint vertexShader, GLuint fragShader, GLuint geometryShader = 0, GLenum inputType = GL_TRIANGLES, GLenum outputType = GL_TRIANGLE_STRIP, int vertOut = 3); 


#endif
