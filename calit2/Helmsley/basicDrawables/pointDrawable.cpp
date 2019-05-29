#include "pointDrawable.h"
#include <stack>
#include <GLES3/gl3.h>
#include <cvrUtil/ARCoreManager.h>
#include <cvrUtil/AndroidHelper.h>


void pointDrawable::Initialization(){
    cvr::glesDrawable::Initialization();
    _shader_program = cvr::assetLoader::instance()->createGLShaderProgramFromFile("shaders/point.vert", "shaders/point.frag");

    _attrib_vertices = glGetAttribLocation(_shader_program,"vPosition");
    _uniform_arMVP_mat =  glGetUniformLocation(_shader_program, "uarMVP");

    //Generate VAO and bind
    glGenVertexArrays(1, &_VAO);
    glBindVertexArray(_VAO);

    //Generate VBO and bind
    glGenBuffers(1, &_VBO);

    //dynamic feed data
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * MAX_POINTS * 4, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(_attrib_vertices);
    glVertexAttribPointer(_attrib_vertices, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(_shader_program);
    glUniform4fv(glGetUniformLocation(_shader_program, "uColor"), 1, _default_color.ptr());
    glUniform1f(glGetUniformLocation(_shader_program, "uPointSize"), _default_size);
    glUseProgram(0);
}

void pointDrawable::drawImplementation(osg::RenderInfo&) const{
    cvr::glStateStack::instance()->PushAllState();

    glUseProgram(_shader_program);
    glUniformMatrix4fv(_uniform_arMVP_mat, 1, GL_FALSE, _mvpMat.ptr());
    glBindVertexArray(_VAO);
    glDrawArrays(GL_POINTS, 0, _point_num);
    glBindVertexArray(0);
    glUseProgram(0);

    cvr::glStateStack::instance()->PopAllState();
}
void pointDrawable::updateOnFrame() {
    _mvpMat = cvr::ARCoreManager::instance()->getMVPMatrix();
    int32_t pre_num = _point_num;

    if(!cvr::ARCoreManager::instance()->getPointCouldData(pointCloudData, _point_num))
        _point_num = pre_num;

    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, _point_num * 4 * sizeof(float), pointCloudData);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
