#ifndef PLUGIN_DCM_RENDERER_H
#define PLUGIN_DCM_RENDERER_H

#include <cvrUtil/glesDrawable.h>
#include <unordered_map>
#include "cubeDrawable.h"

class dcmRenderer: public cubeDrawable{
public:
    void Initialization();
    void updateOnFrame();
    void drawImplementation(osg::RenderInfo&) const;

protected:
    GLuint  _volume_tex_id, _trans_tex_id;
    void assemble_texture_3d();
    void create_trans_texture();
private:
    bool use_raycast = true;
    const float adjustParam_origin[3] = {500.0f, 0.9f, 350.0f};
    float adjustParam[3]= {500.0f, 0.9f, 350.0f};
    glm::vec3 volume_size;
    bool use_color_tranfer = false, use_lighting = false,
         use_interpolation = false, use_simple_cube = false;
};
#endif
