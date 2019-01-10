#include <cvrUtil/AndroidHelper.h>
#include <cvrUtil/AndroidStdio.h>
#include "quadDrawable.h"
#include "pano.h"
using namespace osg;
quadDrawable::quadDrawable(){
    _vertices = new GLfloat[12];
    memcpy(_vertices, _defaultVertices, 12 * sizeof(float));
    _face_id = -1;
}
quadDrawable::quadDrawable(const float *vertices, GLuint id) {
    _vertices = new GLfloat[12];
    memcpy(_vertices, vertices, 12 * sizeof(float));
    _face_id = id;
}
void quadDrawable::Initialization(){
    _shader_program = cvr::assetLoader::instance()->createGLShaderProgramFromFile("shaders/screenquad.vert", "shaders/quad.frag");
    if(!_shader_program)
        LOGE("Failed to create shader program");
    if(!_vertices){
        _vertices = new GLfloat[12];
        memcpy(_vertices, _defaultVertices, 12 * sizeof(GLfloat));
    }
    _attrib_vertices = glGetAttribLocation(_shader_program, "a_Position");
    _attrib_uvs = glGetAttribLocation(_shader_program, "a_TexCoord");


    glGenTextures(0, &_texture_id);
    glBindTexture(GL_TEXTURE_2D, _texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
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
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 8, _uvs, GL_STATIC_DRAW);
    glEnableVertexAttribArray(_attrib_uvs);
    glVertexAttribPointer(_attrib_uvs, 2, GL_FLOAT, GL_FALSE, 2* sizeof(GLfloat), 0);

    glGenBuffers(1, &_EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glUseProgram(_shader_program);
    glUniform1i(glGetUniformLocation(_shader_program, "uSampler"), 0);
    glUseProgram(0);
}


void quadDrawable::drawImplementation(osg::RenderInfo&) const{
    cvr::glStateStack::instance()->PushAllState();
    glUseProgram(_shader_program);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _texture_id);

    int width, height;
//    cvr::ARCoreManager::instance()->getNdkImageSize(width, height);
//    unsigned char* img = cvr::ARCoreManager::instance()->getGrayscaleImageData();
    unsigned char* img =  panoStitcher::instance()->getPanoImageData(width, height);

    if(img)
//        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE,
//                     GL_UNSIGNED_BYTE, img);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img);

    glBindVertexArray(_VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glUseProgram(0);
    cvr::glStateStack::instance()->PopAllState();
}
