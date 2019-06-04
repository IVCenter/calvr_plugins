#include "dcmRenderer_osg.h"
#include <GLES3/gl3.h>
#include <cvrUtil/AndroidHelper.h>
#include <GLES3/gl32.h>
#include "Color.h"
#include <cvrUtil/AndroidDCMHelper.h>
#include <osg/Texture3D>
#include <osg/Texture2D>
#include <cvrUtil/AndroidStdio.h>
#include <osg/ShapeDrawable>
#include <osg/BlendFunc>

using namespace osg;
osg::MatrixTransform* dcmRendererOSG::createDCMRenderer(){
    dcmNode_ = new osg::Geode;
    dcmTrans_ = new osg::MatrixTransform;
    dcmTrans_->addChild(dcmNode_);
    initialization();
    dcmTrans_->setMatrix(osg::Matrixf::scale(scales_.x(), scales_.z(), scales_.y()));
    return dcmTrans_;
}
void dcmRendererOSG::assemble_texture_3d(){
    osg::ref_ptr<osg::Texture3D> volume_tex = new osg::Texture3D;
    ref_ptr<Image> volume_img = new Image;
    volume_img->setImage(
            (int)volume_size.x(), (int)volume_size.y(), (int)volume_size.z(),
            GL_R8, GL_RED, GL_UNSIGNED_BYTE,
            DCMI::volume_data, osg::Image::USE_NEW_DELETE);
//    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//    volume_img->setPacking(1);
//    volume_tex->setDataVariance(osg::Object::STATIC);
    volume_tex->setWrap(osg::Texture3D::WRAP_R, osg::Texture3D::CLAMP_TO_EDGE);
    volume_tex->setWrap(osg::Texture3D::WRAP_T, osg::Texture3D::CLAMP_TO_EDGE);
    volume_tex->setWrap(osg::Texture3D::WRAP_S, osg::Texture3D::CLAMP_TO_EDGE);

    volume_tex->setFilter(osg::Texture3D::MIN_FILTER, osg::Texture3D::LINEAR);
    volume_tex->setFilter(osg::Texture3D::MAG_FILTER, osg::Texture3D::LINEAR);

//    volume_tex->setTextureSize((int)volume_size.x(), (int)volume_size.y(), (int)volume_size.z());
    volume_tex->setImage(volume_img.get());

    ref_ptr<StateSet> stateset = dcmNode_->getOrCreateStateSet();
    stateset->setTextureAttributeAndModes(2, volume_tex.get());
    stateset->addUniform(new osg::Uniform("uSampler_tex", 2));
}
void dcmRendererOSG::create_trans_texture(){
    osg::ref_ptr<osg::Texture2D> trans_tex = new osg::Texture2D;
    ref_ptr<Image> trans_img = new Image;
    trans_img->setImage(
            32, 1, 0,
            GL_RGBA, GL_RGBA, GL_FLOAT,
            (unsigned char*)transfer_color, osg::Image::USE_NEW_DELETE);
//    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    trans_tex->setImage( trans_img.get());
    trans_tex->setWrap( osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_BORDER );
    trans_tex->setWrap( osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_BORDER );
    trans_tex->setFilter( osg::Texture::MIN_FILTER, osg::Texture::LINEAR );
    trans_tex->setFilter( osg::Texture::MAG_FILTER, osg::Texture::LINEAR );

    ref_ptr<StateSet> stateset= dcmNode_->getOrCreateStateSet();
    stateset->setTextureAttributeAndModes(3, trans_tex.get());
    stateset->addUniform(new osg::Uniform("uSampler_trans", 3));
}
void dcmRendererOSG::init_geometry(){
//    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
//    for(int i=0; i<8; i++)
//        vertices->push_back(osg::Vec3f(sVertex[VD_LEN * i], sVertex[VD_LEN * i+1], sVertex[VD_LEN * i+2]));
//
//    geometry = new osg::Geometry();
//    dcmNode_->addDrawable(geometry);
//    geometry->setVertexArray(vertices.get());
//    geometry->addPrimitiveSet(new DrawElementsUInt(GL_TRIANGLES, 36, sIndices));
//    geometry->setUseVertexBufferObjects(true);
//    geometry->setUseDisplayList(false);


    osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable();
    shape->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );

    dcmNode_->addDrawable(shape.get());
    shape->setShape(new osg::Box);

}
void dcmRendererOSG::init_program() {
    ref_ptr<Program> program;
    program = cvr::assetLoader::instance()->createShaderProgramFromFile("shaders/dcmOSG.vert","shaders/dcmOSG.frag");

    if(!program)
        LOGE("===FAILED TO CREATE SHADEER====");
    osg::StateSet * stateSet = dcmNode_->getOrCreateStateSet();
    stateSet->setAttributeAndModes(program.get());

    stateSet->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
    stateSet->setAttributeAndModes(program,osg::StateAttribute::ON| osg::StateAttribute::OVERRIDE);
    stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON| osg::StateAttribute::OVERRIDE);
    stateSet->setRenderBinDetails(500, "transparent");
    stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    osg::BlendFunc *func = new osg::BlendFunc();
    func->setFunction(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    stateSet->setAttributeAndModes(func, osg::StateAttribute::ON| osg::StateAttribute::OVERRIDE);

    //setup textures
    assemble_texture_3d();
    create_trans_texture();

    //frame-changed uniforms
    stateSet->addUniform(new Uniform("uVolume_size", volume_size));
    Uniform * projUniform = new Uniform(Uniform::FLOAT_MAT4, "uProjMat");
    projUniform->setUpdateCallback(new cvr::projMatrixCallback);
    stateSet->addUniform(projUniform);

    Uniform * viewUniform = new Uniform(Uniform::FLOAT_MAT4, "uViewMat");
    viewUniform->setUpdateCallback(new cvr::viewMatrixCallback);
    stateSet->addUniform(viewUniform);

    //toggled uniforms
    toggleUniforms_[U_MODEL_MAT4] = new Uniform("uModelMat", modelMat_);
    toggleUniforms_[U_LIGHT_AMBIENT_VEC3] = new Uniform("Light.Ia", lightIa);
    toggleUniforms_[U_LIGHT_DIFFUSE_VEC3] = new Uniform("Light.Id", lightId);
    toggleUniforms_[U_LIGHT_SPECULAR_VEC3] = new Uniform("Light.Is",lightIs);
    toggleUniforms_[U_LIGHT_SHINESS_F] = new Uniform("shiness", shiness);
    toggleUniforms_[U_SAMPLE_STEP_F] = new Uniform("sample_step_inverse", adjustParam_ori[U_SAMPLE_STEP_F-PARAM_START]);
    toggleUniforms_[U_THRESHOLD_F] = new Uniform("val_threshold", adjustParam_ori[U_THRESHOLD_F-PARAM_START]);
    toggleUniforms_[U_BRIGHTNESS_F] = new Uniform("brightness",  adjustParam_ori[U_BRIGHTNESS_F-PARAM_START]);
    toggleUniforms_[U_OPACITY_THRESHOLD_F] = new Uniform("OpacityThreshold", adjustParam_ori[U_OPACITY_THRESHOLD_F-PARAM_START]);
    toggleUniforms_[U_USE_LIGHTING_B] = new Uniform("u_use_ligting", false);

    for (const auto& uniform : toggleUniforms_)
        stateSet->addUniform(uniform.second);
}
void dcmRendererOSG::initialization(){
    volume_size = osg::Vec3f(DCMI::img_width,
                            DCMI::img_height,
                            DCMI::img_nums);
    modelMat_ = osg::Matrixf::scale(scales_);
    init_geometry();
    init_program();
}

void dcmRendererOSG::setPosition(osg::Matrixf model_mat){
    modelMat_ = osg::Matrixf::scale(scales_) * model_mat;
    toggleUniforms_[U_MODEL_MAT4]->set(modelMat_);
    toggleUniforms_[U_MODEL_MAT4]->dirty();
    osg::Matrixf dcm_modelMat_osg;
    cvr::ARCoreManager::instance()->getLatestHitAnchorModelMat(dcm_modelMat_osg);

    dcmTrans_->setMatrix(dcm_modelMat_osg * osg::Matrixf::scale(scales_.x(), scales_.z(), scales_.y()));
    dcmTrans_->dirtyBound();
}
void dcmRendererOSG::setTuneParameter(int idx, float value){
    toggleUniforms_[(UNIFORM_TYPE)(PARAM_START+idx)]->set(value);
    toggleUniforms_[(UNIFORM_TYPE)(PARAM_START+idx)]->dirty();
}
