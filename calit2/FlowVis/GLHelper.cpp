#include "GLHelper.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <iostream>
#include <cstdlib>

#ifdef WIN32
#include <io.h>
#define open _open
#define read _read
#define close _close
#endif

bool checkShader(GLuint shader, std::string name)
{
    GLchar *log = NULL;
    GLint   val = 0;
    GLint   len = 0;

    /* Check the shader compile status.  If failed, print the log. */

    glGetShaderiv(shader, GL_COMPILE_STATUS,  &val);
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);

    if (val == 0)
    {
        if ((log = (GLchar *) calloc(len + 1, 1)))
        {
            glGetShaderInfoLog(shader, len, NULL, log);

	    std::cerr << "Shader: " << name << std::endl;
            std::cerr << "OpenGL Shader Error:" << std::endl << log;
            free(log);
        }
        return false;
    }
    else
    {
	if ((log = (GLchar *) calloc(len + 1, 1)))
	{
	    glGetShaderInfoLog(shader, len, NULL, log);

	    //std::cerr << "OpenGL Shader Log:" << std::endl << log;
	    free(log);
	}
    }
    return true;
}

bool checkProgram(GLuint program)
{
    GLchar *log = NULL;
    GLint   val = 0;
    GLint   len = 0;

    /* Check the program link status.  If failed, print the log. */

    glGetProgramiv(program, GL_LINK_STATUS,     &val);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);

    if (val == 0)
    {
        if ((log = (GLchar *) calloc(len + 1, 1)))
        {
            glGetProgramInfoLog(program, len, NULL, log);

            std::cerr << "OpenGL Program Error:" << std::endl << log;
            free(log);
        }
        return false;
    }
    return true;
}

bool checkFramebuffer()
{
    const char *s = NULL;

    switch (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT))
    {
    case    GL_FRAMEBUFFER_COMPLETE_EXT:
        return true;

    case    GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
        s = "Framebuffer incomplete attachment";
        break;
    case    GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
        s = "Framebuffer incomplete missing attachment";
        break;
    case    GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
        s = "Framebuffer incomplete dimensions";
        break;
    case    GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
        s = "Framebuffer incomplete formats";
        break;
    case    GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
        s = "Framebuffer incomplete draw buffer";
        break;
    case    GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
        s = "Framebuffer incomplete read buffer";
        break;
    case    GL_FRAMEBUFFER_UNSUPPORTED_EXT:
        s = "Framebuffer unsupported";
        break;
    default:
        s = "Framebuffer error";
        break;
    }

    std::cerr << "OpenGL Error: " << s << std::endl;

    return false;
}

bool checkGL()
{
    const char *s = NULL;

    switch (glGetError())
    {
    case  GL_NO_ERROR:
        return true;

    case  GL_INVALID_ENUM:
        s = "Invalid enumerant";
        break;
    case  GL_INVALID_VALUE:
        s = "Invalid value";
        break;
    case  GL_INVALID_OPERATION:
        s = "Invalid operation";
        break;
    case  GL_STACK_OVERFLOW:
        s = "Stack overflow";
        break;
    case  GL_OUT_OF_MEMORY:
        s = "Out of memory";
        break;
    case  GL_TABLE_TOO_LARGE:
        s = "Table too large";
        break;
    default:
        s = "Unknown";
        break;
    }

    std::cerr << "OpenGL Error: " << s << std::endl;

    return false;
}

bool createShader(std::string file, GLenum shaderType, GLuint & shader)
{
    struct stat st;
    if(stat(file.c_str(),&st) != 0)
    {
	std::cerr << "Error stating shader file: " << file << std::endl;
	return false;
    }

    char * fileBuffer;
    int filefd;
    filefd = open(file.c_str(),O_RDONLY);
    if(!filefd)
    {
	std::cerr << "Error opening shader file: " << file << std::endl;
	return false;
    }

    fileBuffer = new char[st.st_size+1];
    fileBuffer[st.st_size] = '\0';
    read(filefd,fileBuffer,st.st_size);

    close(filefd);

    shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, (const GLchar **) &fileBuffer, NULL);
    glCompileShader(shader);

    delete[] fileBuffer;

    //std::cerr << "Shader: " << file << std::endl;

    return checkShader(shader,file);
}

bool createShaderFromSrc(std::string & source, GLenum shaderType, GLuint & shader, std::string name)
{
    shader = glCreateShader(shaderType);
    const char * srcPtr = source.c_str();
    glShaderSource(shader, 1, (const GLchar **) &srcPtr, NULL);
    glCompileShader(shader);
    
    //std::cerr << "Shader: " << name << std::endl;

    return checkShader(shader,name);
}

bool createProgram(GLuint & program, GLuint vertexShader, GLuint fragShader, GLuint geometryShader, GLenum inputType, GLenum outputType, int vertOut)
{
    program = glCreateProgram();
    if(vertexShader)
    {
	glAttachShader(program,vertexShader);
    }

    if(fragShader)
    {
	glAttachShader(program,fragShader);
    }

    if(geometryShader)
    {
	glAttachShader(program,geometryShader);
	glProgramParameteriEXT(program, GL_GEOMETRY_INPUT_TYPE_EXT, inputType);
	glProgramParameteriEXT(program, GL_GEOMETRY_OUTPUT_TYPE_EXT, outputType);
	glProgramParameteriEXT(program,GL_GEOMETRY_VERTICES_OUT_EXT,vertOut);
    }

    glLinkProgram(program);

    return checkProgram(program);
}
