#ifndef PLUGIN_DCM_RENDERER_OSG_H
#define PLUGIN_DCM_RENDERER_OSG_H

#include <cvrUtil/glesDrawable.h>
#include <osg/MatrixTransform>
#include <glm/gtc/type_ptr.hpp>
#include <unordered_map>
namespace{
    enum UNIFORM_TYPE{
        U_MODEL_MAT4 = 0,

        U_LIGHT_AMBIENT_VEC3,
        U_LIGHT_DIFFUSE_VEC3,
        U_LIGHT_SPECULAR_VEC3,
        U_LIGHT_SHINESS_F,

        U_SAMPLE_STEP_F,
        U_THRESHOLD_F,
        U_BRIGHTNESS_F,
        U_OPACITY_THRESHOLD_F,

        U_USE_LIGHTING_B,
    };
}
class dcmRendererOSG{
public:
    osg::MatrixTransform* createDCMRenderer();
    osg::Geode* getRoot(){return dcmNode_;}
    void setPosition(osg::Matrixf model_mat);
    void setTuneParameter(int idx, float value);

    void Update(){
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_3D, _volume_tex_id);
    }

protected:
    void initialization();
private:
    osg::MatrixTransform* dcmTrans_;
    osg::Geode* dcmNode_;
    osg::Matrixf modelMat_;
    osg::Geometry* geometry;

    GLuint  _volume_tex_id;//MIAODE WOYE BUZHIDAO WEISHENME ZHEGE WORKAAAAAA
    const GLenum _special_unique_texture = GL_TEXTURE3;
    osg::Vec3f volume_size;

    //changeable uniform
    std::unordered_map<UNIFORM_TYPE, osg::Uniform*> toggleUniforms_;
    //adjustable
    osg::Vec3f lightIa = osg::Vec3( 0.8,0.8,0.8 );
    osg::Vec3f lightId = osg::Vec3(  0.7,0.7,0.7 );
    osg::Vec3f lightIs = osg::Vec3(  0.5,0.5,0.5 );
    float shiness = 32.0f;
    const UNIFORM_TYPE PARAM_START = U_SAMPLE_STEP_F;
    float adjustParam_ori[4]= {500.0f, 0.9f, 350.0f, 0.3f};
    osg::Vec3f scales_ = osg::Vec3f(0.1f, -0.1f, 0.05f);//opengl coordinates!!

    //functions
    void init_program();
    void assemble_texture_3d();
    void create_trans_texture();
    void init_geometry();
};
#endif
