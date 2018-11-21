#ifndef PLUGIN_QUADDRAWABLE_H
#define PLUGIN_QUADDRAWABLE_H

#include <cvrUtil/glesDrawable.h>

class quadDrawable: public cvr::glesDrawable {
private:
//    osg::ref_ptr<osg::Geode> qnode;
    GLuint _texture_id;

    GLuint _shader_program;

    GLuint _attrib_vertices;
    GLuint _attrib_uvs;

    GLuint _VAO;
    GLuint _VBO[2];

    const GLfloat _vertices[12] = {
            -1.0f, -1.0f, 0.0f, +1.0f, -1.0f, 0.0f,
            -1.0f, +1.0f, 0.0f, +1.0f, +1.0f, 0.0f,
    };

public:
    void Initialization();
    void updateOnFrame(const float * new_uvs);
    void drawImplementation(osg::RenderInfo &) const;
};
#endif
