#ifndef PLUGIN_PLANEDRAWABLE_H
#define PLUGIN_PLANEDRAWABLE_H

#include <cvrUtil/glesDrawable.h>
#include <cvrUtil/ARCoreManager.h>
#define MAX_PLANE_VERTICES 100

class planeDrawable: public cvr::glesDrawable {
private:
    std::vector<osg::Vec3f> _vertices;
    std::vector<GLushort> _triangles;
    int32_t _vertices_num;
    osg::Vec3f _color;
    GLuint _VAO;
    GLuint _VBO;
    GLuint _EBO;
    float* raw_vertices;
    osg::Matrixf _model_mat;
    osg::Matrixf _view_proj_mat;
    osg::Vec3f _normal_vec;

    GLuint _texture_id;
    GLuint _shader_program;

    GLint _attrib_vertices;
    GLint _uniform_mvp_mat;
    GLint _uniform_tex_sampler;
    GLint _uniform_normal_vec;
    GLint _uniform_model_mat;
    GLint _uniform_color;

    void _update_plane_vertices();
public:
    void updateOnFrame(ArPlane* plane, osg::Vec3f color);
    void Initialization();
    void drawImplementation(osg::RenderInfo&) const;
    void Reset(){}
};


#endif
