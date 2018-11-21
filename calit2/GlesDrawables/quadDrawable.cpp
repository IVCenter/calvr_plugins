#include <cvrUtil/AndroidHelper.h>
#include <osg/Texture2D>
#include <cvrUtil/AndroidStdio.h>
#include "quadDrawable.h"
#define HEIGHT_RATE 1.0f
using namespace osg;
void quadDrawable::Initialization(){
    _shader_program = cvr::assetLoader::instance()->createGLShaderProgramFromFile("shaders/screenquad.vert", "shaders/quad.frag");
    if(!_shader_program)
        LOGE("Failed to create shader program");

    _attrib_vertices = glGetAttribLocation(_shader_program, "a_Position");
    _attrib_uvs = glGetAttribLocation(_shader_program, "a_TexCoord");


    glGenTextures(0, &_texture_id);
    glBindTexture(GL_TEXTURE_2D, _texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    //Generate VAO and bind
    glGenVertexArrays(1, &_VAO);
    glBindVertexArray(_VAO);

    //Generate VBO and bind
    glGenBuffers(2, _VBO);

    glBindBuffer(GL_ARRAY_BUFFER, _VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 12, _vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(_attrib_vertices);
    glVertexAttribPointer(_attrib_vertices, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat),0);

    glBindBuffer(GL_ARRAY_BUFFER, _VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 8, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(_attrib_uvs);
    glVertexAttribPointer(_attrib_uvs, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(_shader_program);
    glUniform1i(glGetUniformLocation(_shader_program, "uSampler"), 0);
    glUseProgram(0);
}

void quadDrawable::updateOnFrame(const float * new_uvs){
    glBindBuffer(GL_ARRAY_BUFFER, _VBO[1]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 8* sizeof(float), new_uvs);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void quadDrawable::drawImplementation(osg::RenderInfo&) const{
    cvr::glStateStack::instance()->PushAllState();
    glUseProgram(_shader_program);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _texture_id);

    uint8_t * img = cvr::ARCoreManager::instance()->getImageData();
    if(img)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, img);

    glBindVertexArray(_VAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glUseProgram(0);
    cvr::glStateStack::instance()->PopAllState();
}
