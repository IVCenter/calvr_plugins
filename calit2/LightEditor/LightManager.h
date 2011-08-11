/** Light Manager Class used to handle the different lights created
    during the lifetime of the plugin
*/

#ifndef _LIGHT_MANAGER_H
#define _LIGHT_MANAGER_H

// Std 
#include <iostream>
#include <list>
#include <stack>
#include <string>

// OSG Headers
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osg/StateSet>
#include <osg/Switch>
#include <osg/MatrixTransform>

class LightManager
{
   public:
      LightManager();
      ~LightManager();
      static const int VERSION = 1;

      class LightBundle
      {
         public:
            LightBundle();
            osg::ref_ptr<osg::Switch> graphicModelSwitch;
            osg::ref_ptr<osg::MatrixTransform> graphicTrans;
            osg::ref_ptr<osg::Light> light;
            osg::ref_ptr<osg::LightSource> lightSource;
            osg::ref_ptr<osg::Shape> modelShape;
            osg::ref_ptr<osg::Geode> modelGeode;
            osg::ref_ptr<osg::Material> modelMaterial;
	    int num;
            bool on;
            std::string name;
            float lastValidCutoff;
      };

      class LightInfo
      {
         public:
            LightInfo(osg::Vec4 position, osg::Vec4 ambient, osg::Vec4 diffuse, osg::Vec4 specular,
               float constant, float linear, float quadratic,
               osg::Vec3 spotDirection, float spotExponent, float spotCutoff,
               std::string name, bool on);
            ~LightInfo();

            osg::Vec4 position;
            osg::Vec4 ambient;
            osg::Vec4 diffuse;
            osg::Vec4 specular;
            float constant;
            float linear;
            float quadratic;
            osg::Vec3 spotDirection;
            float spotExponent;
            float spotCutoff;
            std::string name;
            bool on;
      };

      enum Type { SPOT, POINT, DIRECTIONAL };

      // Light properties
      void LightNum(const int num);
      int  LightNum();
      void Ambient(const osg::Vec4 v4);
      osg::Vec4 Ambient();
      void Diffuse(const osg::Vec4 v4);
      osg::Vec4 Diffuse();
      void Specular(const osg::Vec4 v4);
      osg::Vec4 Specular();
      void Position(const osg::Vec4 v4);
      osg::Vec4 Position();
      void SpotDirection(const osg::Vec3 v3);
      osg::Vec3 SpotDirection();
      void SpotExponent(const float exp);
      float SpotExponent();
      void SpotCutoff(const float exp);
      float SpotCutoff();
      void ConstantAttenuation(const float f);
      float ConstantAttenuation();
      void LinearAttenuation(const float f);
      float LinearAttenuation();
      void QuadraticAttenuation(const float f);
      float QuadraticAttenuation();
      // End Light properties
      void PhysicalPosition(const osg::Vec4 v4);
      osg::Vec4 PhysicalPosition();
      void LightType(const Type type);
      Type LightType();
      bool LightOn();

      void storeSelectedLight();
      void recallStoredLight();

      void createNewLight();
      bool isLightSelected(bool expected = false); // expected = true -> cerr on failure
      bool selectLightByName(std::string name);
      bool selectLightByGeodePtr(osg::Geode * geodePtr);
      bool enableLight();
      void disableLight();
      bool isEnabled();

      int getNumLightsEnabled();
      void populateLightInfoList(std::list<LightInfo*> &liList, bool onlyEnabled = true);
      void populateLightNameList(std::list<std::string> &lnList);

      // Handles for Graphic Models
      void enableGraphicModels();
      void disableGraphicModels();
      void toggleGraphicModels();

      void GraphicSize(float size);
      float GraphicSize();
      
      // Light Bundle Properties
      std::string Name();
      void Name(const std::string n);
      void Name(const char * n);
      // End Light Bundle Properties
      
      bool resetLightsChanged();

   private:
      static const int MAX_LIGHTS = 32;

      float graphicSize;
      osg::ref_ptr<osg::Group> lightGroup;
      osg::ref_ptr<osg::StateSet> lightSS;
      std::list<LightBundle*> mLights;
      bool lightInUse[MAX_LIGHTS];
      bool graphicsEnabled;
      std::stack<LightBundle*> selectedLight;
      unsigned int enabledLights;
      bool lightsChanged;

      void initBundle(LightBundle *lb);
      void changeGraphicShape();
      void updateGraphicModels();
      void updateGraphicTrans();      
};

#endif
