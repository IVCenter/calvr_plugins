#ifndef PLUGIN_QUADDRAWABLE_H
#define PLUGIN_QUADDRAWABLE_H

#include <cvrUtil/glesDrawable.h>

class quadDrawable: public cvr::glesDrawable {
private:
    GLuint _texture_id;
    GLuint _face_id;

    GLuint _shader_program;

    GLuint _attrib_vertices;
    GLuint _attrib_uvs;

    GLuint _VAO;
    GLuint _VBO[2];
    GLuint _EBO;

    GLfloat *_vertices = nullptr;
    const GLfloat _defaultVertices[12] = {-1.0f, -1.0f, 0.0f, -1.0f, +1.0f, 0.0f, +1.0f, -1.0f, 0.0f, +1.0f, +1.0f, 0.0f};
    const GLfloat _uvs[8]={0.0f, 0.0f, 1.0f, 0.0f,1.0f, 1.0f, 0.0f, 1.0f};
    const GLuint elements[6] = {0, 1, 2, 2, 3, 0};

    int _image_width=0, _image_height=0;

public:
    quadDrawable();
    quadDrawable(const float* vertices, GLuint id);
    void Initialization();
//    void updateOnFrame(const float * new_uvs);
    void drawImplementation(osg::RenderInfo &) const;
};
#endif
