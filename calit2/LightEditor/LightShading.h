#ifndef _LIGHT_SHADING_H
#define _LIGHT_SHADING_H

#include <iostream>
#include <list>
#include <string>

#include <osg/Program>
#include <osg/Shader>
#include <osg/Image>
#include <osg/TextureRectangle>

#include "LightManager.h"

#define MAX_LIGHTS 16
#define NUM_OF_ATTRIBUTES 6

class LightShading
{
  public:
      LightShading(LightManager* lm);
      ~LightShading();   
      void UpdateUniforms();
       
 private:
      LightManager* mLightManager;  

      osg::ref_ptr<osg::Program>  mLightingProgram;
      osg::ref_ptr<osg::Shader>   mLightingVert;
      osg::ref_ptr<osg::Shader>   mLightingFrag;
      osg::ref_ptr<osg::StateSet> mlightingState;

      osg::ref_ptr<osg::Uniform> lightTex;
      osg::ref_ptr<osg::TextureRectangle> lightTexture;
      osg::ref_ptr<osg::Image> mLightData;  

      int rtWidth, rtHeight; 

     
};

#endif
