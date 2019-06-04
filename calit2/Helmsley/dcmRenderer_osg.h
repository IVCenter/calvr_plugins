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

    void Update(){}

protected:
    void initialization();
private:
    osg::MatrixTransform* dcmTrans_;
    osg::Geode* dcmNode_;
    osg::Matrixf modelMat_;
    osg::Geometry* geometry;


    const int VD_LEN = 3;
    const GLfloat sVertex[24] = {
            -0.5f,-0.5f,0.5f,	//x0, y0, z1, //	//v0
            0.5f,-0.5f,0.5f,	//x1,y0,z1, //	//v1
            0.5f,0.5f,0.5f,	 //x1, y1, z1,//	//v2
            -0.5f,0.5f,0.5f, //x0,y1,z1, //	//v3
            -0.5f,-0.5f,-0.5f,	//x0,y0,z0,//	//v4
            0.5f,-0.5f,-0.5f,	//x1,y0,z0,//	//v5
            0.5f,0.5f,-0.5f,	//x1,y1,z0, //	//v6
            -0.5f,0.5f,-0.5f //x0,y1,z0//	//v7
    };
    const GLuint sIndices[36] = { 0,1,2,0,2,3,	//front
                                  4,6,7,4,5,6,	//back
                                  4,0,3,4,3,7,	//left
                                  1,5,6,1,6,2,	//right
                                  3,2,6,3,6,7,	//top
                                  4,5,1,4,1,0,	//bottom
    };
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
//    float adjustParam[4]= {500.0f, 0.9f, 350.0f, 0.3f};
    osg::Vec3f scales_ = osg::Vec3f(0.1f, -0.1f, 0.05f);//opengl coordinates!!

    //functions
    void init_program();
    void assemble_texture_3d();
    void create_trans_texture();
    void init_geometry();
};
#endif
