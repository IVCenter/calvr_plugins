#ifndef PLUGIN_POINTDRAWABLE_H
#define PLUGIN_POINTDRAWABLE_H

#include <osg/Drawable>
#include <cvrUtil/glesDrawable.h>

#define MAX_POINTS 100
class pointDrawable: public cvr::glesDrawable {
private:
    GLuint _VAO;
    GLuint _VBO;
    GLuint _shader_program;

    GLuint _attrib_vertices;
    GLuint _uniform_arMVP_mat;

    osg::Vec4f _default_color= osg::Vec4f(1.0, 0.5, 0.0, 1.0);
    float _default_size = 20.0f;
    osg::Matrixf _mvpMat;

    int32_t _point_num = 0;
    float *pointCloudData;

public:
    void Initialization();
    void updateOnFrame();
    void drawImplementation(osg::RenderInfo&) const;
};

#endif //MYGLES_POINTDRAWABLE_H
