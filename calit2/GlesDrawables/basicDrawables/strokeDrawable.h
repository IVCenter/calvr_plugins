#ifndef PLUGIN_STROKE_DRAWABLE_H
#define PLUGIN_STROKE_DRAWABLE_H

#include <cvrUtil/glesDrawable.h>

class strokeDrawable: public cvr::glesDrawable {
private:
    bool _onscrren = false;
    GLuint _VAO;
    GLuint _VBO[2];
    GLuint _shader_program;

    GLuint _attrib_vertices_screen;
    GLuint _attrib_vertices;
    GLuint _attrib_offsets;
    GLuint _uniform_view;
    GLuint _uniform_proj;

    osg::Vec4f _default_color= osg::Vec4f(1.0, 0.5, 0.0, 1.0);
    float _default_size = 30.0f;
    float _default_line_width = 10.0f;
    osg::Matrixf _viewMat, _projMat;

//    float last_strokeData[8];
    float strokeData[8] = {.0};
    float offsetData[4] = {.0};
    float screenData[4] = {.0};

public:
    osg::ref_ptr<osg::Geode> createDrawableNode(float startx, float starty){
        _onscrren = true;
        screenData[0] = startx; screenData[1] = starty;
        glNode = cvr::glesDrawable::createDrawableNode();
        return glNode.get();
    }

    void updateOnFrame(osg::Vec3f from, osg::Vec3f to, osg::Vec2f offset);

    void updateOnFrame(osg::Vec3f to);

    void Initialization();

    void drawImplementation(osg::RenderInfo&) const;

    void Reset(){}
};

#endif
