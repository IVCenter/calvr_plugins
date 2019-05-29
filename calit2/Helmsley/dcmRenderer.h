#ifndef PLUGIN_DCM_RENDERER_H
#define PLUGIN_DCM_RENDERER_H

#include <cvrUtil/glesDrawable.h>
#include <unordered_map>
#include "cubeDrawable.h"
#include <glm/gtc/type_ptr.hpp>
class dcmRenderer: public cubeDrawable{
public:
    void Initialization();
    void updateOnFrame();
    void drawImplementation(osg::RenderInfo&) const;

    void setPosition(osg::Matrixf model_mat){
        _modelMat = glm::make_mat4(model_mat.ptr()) * _scaleMat;
    }
protected:
    GLuint  _volume_tex_id, _trans_tex_id;
    void assemble_texture_3d();
    void create_trans_texture();
private:
    bool use_raycast = true;
    const float adjustParam_origin[3] = {500.0f, 0.9f, 350.0f};
    float adjustParam[3]= {500.0f, 0.9f, 350.0f};
    glm::mat4 _scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, -0.5f, 0.25f));
    glm::vec3 volume_size;
    bool use_color_tranfer = false, use_lighting = false,
         use_interpolation = false, use_simple_cube = false;
};
#endif
