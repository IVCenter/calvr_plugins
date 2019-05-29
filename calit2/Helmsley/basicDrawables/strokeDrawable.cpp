#include "strokeDrawable.h"
#include <GLES3/gl3.h>
#include <cvrUtil/AndroidHelper.h>

void strokeDrawable::Initialization() {
    cvr::glesDrawable::Initialization();

    if(_onscrren){
        _shader_program = cvr::assetLoader::instance()->createGLShaderProgramFromFile("shaders/stroke_screen.vert", "shaders/point.frag");

        _attrib_vertices_screen = glGetAttribLocation(_shader_program,"vPositionScreen");
    }else{
        _shader_program = cvr::assetLoader::instance()->createGLShaderProgramFromFile("shaders/stroke.vert", "shaders/point.frag");

        _attrib_vertices = glGetAttribLocation(_shader_program,"vPosition");
        _attrib_offsets = glGetAttribLocation(_shader_program, "vOffset");

        _uniform_view =  glGetUniformLocation(_shader_program, "uView");
        _uniform_proj =  glGetUniformLocation(_shader_program, "uProj");
    }


    //Generate VAO and bind
    glGenVertexArrays(1, &_VAO);
    glBindVertexArray(_VAO);

    if(_onscrren){
        glGenBuffers(1, _VBO);
        glBindBuffer(GL_ARRAY_BUFFER, _VBO[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 2 * 2, nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(_attrib_vertices_screen);
        glVertexAttribPointer(_attrib_vertices_screen, 2, GL_FLOAT, GL_FALSE, 0, 0);
    }else{
        //Generate VBO and bind
        glGenBuffers(2, _VBO);
        //dynamic feed data
        glBindBuffer(GL_ARRAY_BUFFER, _VBO[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 2 * 4, nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(_attrib_vertices);
        glVertexAttribPointer(_attrib_vertices, 4, GL_FLOAT, GL_FALSE, 0, 0);

        //dynamic feed data
        glBindBuffer(GL_ARRAY_BUFFER, _VBO[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 2 * 2, nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(_attrib_offsets);
        glVertexAttribPointer(_attrib_offsets, 2, GL_FLOAT, GL_FALSE, 0, 0);
    }


    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(_shader_program);
    glUniform1f(glGetUniformLocation(_shader_program, "uPointSize"), _default_size);
    glUniform4fv(glGetUniformLocation(_shader_program, "uColor"), 1, _default_color.ptr());
    glUseProgram(0);
}

void strokeDrawable::drawImplementation(osg::RenderInfo &) const {
    cvr::glStateStack::instance()->PushAllState();


    glUseProgram(_shader_program);
    if(!_onscrren) {
        glUniformMatrix4fv(_uniform_view, 1, GL_FALSE, _viewMat.ptr());
        glUniformMatrix4fv(_uniform_proj, 1, GL_FALSE, _projMat.ptr());
    }
    glLineWidth(_default_line_width);
    glBindVertexArray(_VAO);
    glDrawArrays(GL_LINES, 0, 2);
    glDrawArrays(GL_POINTS, 0, 2);
    glBindVertexArray(0);

    glUseProgram(0);
    cvr::glStateStack::instance()->PopAllState();
}
void strokeDrawable::updateOnFrame(osg::Vec3f from, osg::Vec3f to, osg::Vec2f offset) {
    osg::Vec3f fromPose = osg::Vec3f(from.x(), from.z(), -from.y());
    float *start_pos = fromPose.ptr();
    float *end_pos = to.ptr();

    _viewMat = *cvr::ARCoreManager::instance()->getViewMatrix();
    _projMat = *cvr::ARCoreManager::instance()->getProjMatrix();

    for(int i=0;i<3;i++){
        strokeData[i] = start_pos[i];
        strokeData[4 + i] = end_pos[i];
    }
    strokeData[3] = 1.0; strokeData[7] = 1.0f;

    offsetData[0] = offset.x(); offsetData[1] = offset.y();

    glBindBuffer(GL_ARRAY_BUFFER, _VBO[0]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 8 * sizeof(float), strokeData);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, _VBO[1]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * sizeof(float), offsetData);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void strokeDrawable::updateOnFrame(osg::Vec3f to) {
    osg::Vec4f toPose = osg::Vec4f(to.x(), to.z(), -to.y(), 1.0);
    toPose = toPose* cvr::ARCoreManager::instance()->getMVPMatrix();
    screenData[2] = toPose.x()/toPose.w();
    screenData[3] = toPose.y()/toPose.w();

    glBindBuffer(GL_ARRAY_BUFFER, _VBO[0]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * sizeof(float), screenData);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}