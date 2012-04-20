#include <osgDB/FileUtils>
#include <osg/BlendEquation>

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>

#include <stdio.h>
#include <math.h>

#include "LightShading.h"
#include "Utility.h"

/////////////////////////////////////////////////////////////////////////
// Const / Deconst

LightShading::LightShading(LightManager* lm)
{
   // Set reference to the light manager
   mLightManager = lm;

   // Load shader sources
   //osg::setNotifyLevel(osg::DEBUG_INFO);
   mlightingState = cvr::PluginHelper::getObjectsRoot()->getOrCreateStateSet();  
   mLightingProgram = new osg::Program;
   mLightingVert = new osg::Shader( osg::Shader::VERTEX );
   mLightingFrag = new osg::Shader( osg::Shader::FRAGMENT );
   mLightingProgram->addShader( mLightingFrag );  
   mLightingProgram->addShader( mLightingVert );  

   if (!utl::loadShaderSource( mLightingVert, 
           cvr::ConfigManager::getEntry("vert","Plugin.LightEditor.LightShader","") ))
      std::cerr << "Couldn't load vertex shader!\n";
   if (!utl::loadShaderSource( mLightingFrag, 
           cvr::ConfigManager::getEntry("frag","Plugin.LightEditor.LightShader","") ))
      std::cerr << "Couldn't load fragment shader!\n";  
   
   mlightingState->setAttributeAndModes( mLightingProgram, osg::StateAttribute::ON ); 
   //osg::setNotifyLevel(osg::NOTICE);
   
   // Need to generate texture on first run
   mLightData = new osg::Image;
   mLightData->allocateImage(NUM_OF_ATTRIBUTES, MAX_LIGHTS, 1, GL_RGBA, GL_FLOAT);  

   lightTex = new osg::Uniform("lightTex",
       cvr::ConfigManager::getInt("textID","Plugin.LightEditor.LightShader", 1));         
   mlightingState->addUniform(lightTex);
   
   lightTexture = new osg::TextureRectangle(mLightData);
   lightTexture->setFilter(osg::Texture::MIN_FILTER,osg::Texture::NEAREST);
   lightTexture->setFilter(osg::Texture::MAG_FILTER,osg::Texture::NEAREST);
   lightTexture->setWrap(osg::Texture::WRAP_S,osg::TextureRectangle::CLAMP);
   lightTexture->setWrap(osg::Texture::WRAP_T,osg::TextureRectangle::CLAMP); 
   lightTexture->setInternalFormat(GL_RGBA32F_ARB);
   lightTexture->setResizeNonPowerOfTwoHint(false);  
}

LightShading::~LightShading()
{
}

void LightShading::UpdateUniforms()
{
   bool regenerateTexture = mLightManager->resetLightsChanged();

   // Retrieve enabled lights
   std::list<LightManager::LightInfo*> liList;
   mLightManager->populateLightInfoList(liList);

   std::list<LightManager::LightInfo*>::iterator iter;
   iter = liList.begin();   

   bool noMoreLights = false;
   for (int i = 0; i < MAX_LIGHTS; i++)
   {
      if (!(noMoreLights = iter == liList.end()))
      {
         // Check if texture needs to be generated
         if (regenerateTexture)
         {
            // Reformat texture with new lighting values            
            {         
               ((float*)mLightData->data(0,i))[0] = (*iter)->position.x();
               ((float*)mLightData->data(0,i))[1] = (*iter)->position.y();
               ((float*)mLightData->data(0,i))[2] = (*iter)->position.z();
               ((float*)mLightData->data(0,i))[3] = (*iter)->position.w();               
      
               ((float*)mLightData->data(1,i))[0] = (*iter)->ambient.x();
               ((float*)mLightData->data(1,i))[1] = (*iter)->ambient.y();
               ((float*)mLightData->data(1,i))[2] = (*iter)->ambient.z();
               ((float*)mLightData->data(1,i))[3] = (*iter)->ambient.w();
      
               ((float*)mLightData->data(2,i))[0] = (*iter)->diffuse.x();
               ((float*)mLightData->data(2,i))[1] = (*iter)->diffuse.y();
               ((float*)mLightData->data(2,i))[2] = (*iter)->diffuse.z();
               ((float*)mLightData->data(2,i))[3] = (*iter)->diffuse.w();
      
               ((float*)mLightData->data(3,i))[0] = (*iter)->specular.x();
               ((float*)mLightData->data(3,i))[1] = (*iter)->specular.y();
               ((float*)mLightData->data(3,i))[2] = (*iter)->specular.z();
               ((float*)mLightData->data(3,i))[3] = (*iter)->specular.w();    
      
               ((float*)mLightData->data(4,i))[0] = (*iter)->spotDirection.x();
               ((float*)mLightData->data(4,i))[1] = (*iter)->spotDirection.y();
               ((float*)mLightData->data(4,i))[2] = (*iter)->spotDirection.z();
               ((float*)mLightData->data(4,i))[3] = (*iter)->spotExponent;      
	       
               ((float*)mLightData->data(5,i))[0] = cos((*iter)->spotCutoff*osg::PI/180.0);
               ((float*)mLightData->data(5,i))[1] = (*iter)->constant;
               ((float*)mLightData->data(5,i))[2] = (*iter)->linear;
               ((float*)mLightData->data(5,i))[3] = (*iter)->quadratic;  
            } 
         }

         iter++;
      }
      else 
         //lightType->setElement(i, -1.0f);
         ((float*)mLightData->data(0,i))[3] = -1.0f;
   }   
   
   if (regenerateTexture)
   {       
      mLightData->dirty();        
      mlightingState->setTextureAttributeAndModes(
          cvr::ConfigManager::getInt("textID","Plugin.LightEditor.LightShader", 1),
          lightTexture);          
   }   
}
