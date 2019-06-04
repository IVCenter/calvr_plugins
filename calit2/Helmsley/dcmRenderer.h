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
    void setTuneParameter(int idx, float value){adjustParam[idx] = value;}
protected:
    void assemble_texture_3d();
private:
    float adjustParam[4]= {500.0f, 0.9f, 350.0f, 0.3f};
    glm::mat4 _scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, -0.5f, 0.25f));
    glm::vec3 volume_size;
    bool use_lighting = false,
         use_interpolation = false, use_simple_cube = false;
};
#endif
