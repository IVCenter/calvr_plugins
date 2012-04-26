#include "LightManager.h"

#include <iostream>
#include <sstream>
#include <math.h>

#include <osg/Geometry>
#include <osg/Geode>
#include <osg/PolygonMode>

#include <cvrKernel/PluginHelper.h>

const double rad2deg = 180/osg::PI;
const double deg2rad = osg::PI/180;

LightManager::LightManager()
{
   lightGroup = new osg::Group;
   cvr::PluginHelper::getObjectsRoot()->addChild(lightGroup);
   lightSS = lightGroup->getOrCreateStateSet();
   for (int i=0;i<MAX_LIGHTS;i++) lightInUse[i] = false;
   selectedLight.push(NULL);
   // graphic models default
   graphicsEnabled = true;
   enabledLights = 0;
   graphicSize = 3;
   lightsChanged = false;
}

LightManager::~LightManager()
{
   // deallocate anything in mLights
   std::list<LightBundle*>::iterator i;
   for (i=mLights.begin(); i != mLights.end(); i++)
      delete *i;
}

LightManager::LightBundle::LightBundle()
{
   graphicModelSwitch = new osg::Switch();   
   graphicTrans = new osg::MatrixTransform();
   light = new osg::Light;
   lightSource = new osg::LightSource;
   modelGeode = new osg::Geode();
   modelMaterial = new osg::Material;
   num = -1;
   on = false;
   name.assign("null Light");
   lastValidCutoff = 45.0;
}

LightManager::LightInfo::LightInfo(osg::Vec4 position, osg::Vec4 ambient, osg::Vec4 diffuse,
   osg::Vec4 specular, float constant, float linear, float quadratic,
   osg::Vec3 spotDirection, float spotExponent, float spotCutoff, std::string name, bool on)
{
   this->position      = position;
   this->ambient       = ambient;
   this->diffuse       = diffuse;
   this->specular      = specular;
   this->constant      = constant;
   this->linear        = linear;
   this->quadratic     = quadratic;
   this->spotDirection = spotDirection;
   this->spotExponent  = spotExponent;
   this->spotCutoff    = spotCutoff;
   this->name          = name;
   this->on            = on;
}

void LightManager::storeSelectedLight()
{
   // Increase selectedLight stack size, copying current top as new top
   selectedLight.push(selectedLight.top());
}

void LightManager::recallStoredLight()
{
   // Pop back down the selected light stack, when valid
   if (selectedLight.size() == 1)
   {
      std::cerr<<"Error: Cannot call recallStoredLight if nothing has been stored yet.\n";
      return;
   }

   selectedLight.pop();
}

void LightManager::createNewLight()
{
   static int lightCount;

   LightBundle * newLight = new LightBundle();
   mLights.push_back(newLight);
   selectLight(newLight);

   std::ostringstream os;
   os << "Light " << lightCount++;
   Name(os.str());

   // Setup Actual Light Components

   newLight->lightSource->setLight(newLight->light.get());
   lightGroup->addChild(newLight->lightSource.get());

   enableLight();
   
   // Setup Graphical Light Model

   newLight->graphicModelSwitch->addChild(newLight->modelGeode.get(), graphicsEnabled);  
   newLight->graphicTrans->addChild(newLight->graphicModelSwitch.get());
   lightGroup->addChild(newLight->graphicTrans.get());
   osg::ref_ptr<osg::StateSet> shapeSS = new osg::StateSet;
   shapeSS->setAttribute(newLight->modelMaterial.get());
   newLight->modelGeode->setStateSet(shapeSS.get());

   changeGraphicShape();
   lightsChanged = true;
}

void LightManager::updateGraphicModels()
{
   std::list<LightBundle*>::iterator i;
   for (i = mLights.begin(); i != mLights.end(); i++)
   {
      if (graphicsEnabled)
         (*i)->graphicModelSwitch->setAllChildrenOn();
      else
         (*i)->graphicModelSwitch->setAllChildrenOff();
   }
}

void LightManager::updateGraphicTrans()
{
   if (!isLightSelected( true ))
      return;

   osg::Vec4 pos = Position();
   osg::Vec3 pos3 = osg::Vec3(pos.x(),pos.y(),pos.z());

   osg::Matrix matrix = osg::Matrix::translate(pos3);

   // Rotate directional lights to point towards origin, always
   if (pos.w() == 0) // Directional: Rotate via position
   {
      if (pos.x() != 0.f || pos.y() != 0.f) // need a valid rotation matrix
      {
         matrix = osg::Matrix::rotate(osg::Vec3(0,0,1), pos3 * -1) * matrix;
      }
   }
   else if (SpotCutoff() != 180.0) // Spot: Rotate via direction
   {
      osg::Vec3 dir = SpotDirection();
      if (dir.x() != 0.f || dir.y() != 0.f)
      {
         matrix = osg::Matrix::rotate(osg::Vec3(0,0,1),dir * -1) * matrix;
      }
   }

   selectedLight.top()->graphicTrans->setMatrix(matrix);

}

bool LightManager::resetLightsChanged()
{
   bool wereChanged = lightsChanged;
   lightsChanged = false;
   return wereChanged;
}

void LightManager::enableGraphicModels()
{
   if (graphicsEnabled) return;
   
   graphicsEnabled = true;
   updateGraphicModels();
}

void LightManager::disableGraphicModels()
{
   if (!graphicsEnabled) return;   
   
   graphicsEnabled = false;
   updateGraphicModels();
}

void LightManager::toggleGraphicModels()
{
   graphicsEnabled ? disableGraphicModels() : enableGraphicModels();
}

/* Assigns a number to the light.
 * Returns true on a success, and false if all the light numerals are in use.
 */
bool LightManager::enableLight()
{
   if (!isLightSelected( true ))
      return false;
      
   if (LightOn())
      return true;

   int num=MAX_LIGHTS;

   while (--num >= 0)
      if (!lightInUse[num])
         break;

   if (num<0)
   {
      std::cerr << "LightManager::availableLightNum called with "<<MAX_LIGHTS<<" lights already enabled.\n";
      return false;
   }

   LightNum(num);
   lightInUse[num] = true;

   LightBundle * light = selectedLight.top();

   light->on = true;

   light->lightSource->setLocalStateSetModes(osg::StateAttribute::ON);
   light->lightSource->setStateSetModes(*lightSS,osg::StateAttribute::ON);

   enabledLights++;

   lightsChanged = true;

   return true;
}

void LightManager::disableLight()
{
   if (!isLightSelected( true ))
      return;
      
   if (!LightOn())
      return;

   LightBundle * light = selectedLight.top();

   lightInUse[light->num] = false;
   light->on = false;

   light->lightSource->setLocalStateSetModes(osg::StateAttribute::OFF);
   light->lightSource->setStateSetModes(*lightSS,osg::StateAttribute::OFF);

   lightsChanged = true;

   enabledLights--;
}

void LightManager::selectLight(LightBundle * light)
{
   if (selectedLight.top())
   {
      osg::StateSet * ss = selectedLight.top()->graphicTrans->getOrCreateStateSet();
      osg::PolygonMode * pm = dynamic_cast< osg::PolygonMode * >(ss->getAttribute(osg::StateAttribute::POLYGONMODE));

      if (!pm)
          ss->setAttribute(pm = new osg::PolygonMode);

      pm->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
   }

   selectedLight.top() = light;

   if (selectedLight.top())
   {
      osg::StateSet * ss = selectedLight.top()->graphicTrans->getOrCreateStateSet();
      osg::PolygonMode * pm = dynamic_cast< osg::PolygonMode * >(ss->getAttribute(osg::StateAttribute::POLYGONMODE));

      if (!pm)
          ss->setAttribute(pm = new osg::PolygonMode);

      pm->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
   }
}

void LightManager::changeGraphicShape()
{
   if (!isLightSelected( true ))
      return;

   Type type = LightType();
   float size = GraphicSize();

   LightBundle * light = selectedLight.top();
   float cutoff;
   switch(type)
   {
      case SPOT: // Add .01 to the height to prevent artifacts when rendering.
         if ((cutoff = SpotCutoff()) < 45)
            light->modelShape = new osg::Cone(osg::Vec3(0,0,0),size*tan(cutoff*deg2rad),size + .01);
         else 
            light->modelShape = new osg::Cone(osg::Vec3(0,0,0),size,size/tan(cutoff*deg2rad) + .01);
         break;
      case DIRECTIONAL:
         light->modelShape = new osg::Cylinder(osg::Vec3(0,0,0),size/3,size);
         break;
      default:
         std::cerr<<"LightManager::changeGraphicShape encountered invalid"
         <<"LightType() return of value "<<type<<"\n+ Defaulting to POINT model.\n";
      case POINT:
         light->modelShape = new osg::Sphere(osg::Vec3(0,0,0),size);
         break;
   } 
   
   // Create a shape drawable
   osg::ref_ptr<osg::ShapeDrawable> shapeDrawable = new osg::ShapeDrawable(light->modelShape.get());
   // Don't use display lists so that geometry auto-updates each rendering
   shapeDrawable->setUseDisplayList(false);
   // Remove previous drawable(s) and add in the new one
   light->modelGeode->removeDrawables(0,light->modelGeode->getNumDrawables());
   light->modelGeode->addDrawable(shapeDrawable.get());
}

// expected defaults to false if not supplied
bool LightManager::isLightSelected(bool expected)
{
   if (selectedLight.top() == NULL)
   {
      if (expected)
         std::cerr << "No light selected!\n";

      return false;
   }
   return true;
}

bool LightManager::selectLightByName(std::string name) 
{
   std::list<LightBundle*>::iterator i;
   for (i = mLights.begin(); i != mLights.end(); i++)
   {
      if ((*i)->name == name)
      {
         selectLight(*i);
         return true;
      }
   }
   
   return false;
}

bool LightManager::selectLightByGeodePtr(osg::Geode * geode) 
{
   std::list<LightBundle*>::iterator i;
   for (i = mLights.begin(); i != mLights.end(); i++)
   {
      if ((*i)->modelGeode == geode)
      {
         selectLight(*i);
         return true;
      }
   }
   
   return false;
}

int LightManager::getNumLightsEnabled()
{
   return enabledLights;
}

void LightManager::populateLightInfoList(std::list<LightManager::LightInfo*> &liList, bool onlyEnabled)
{
   std::list<LightBundle*>::iterator i;
   for (i = mLights.begin(); i != mLights.end(); i++)
   {
      selectedLight.push(*i);

      if (!onlyEnabled || LightOn())
      {
         LightInfo* newInfo = new LightInfo(Position(), Ambient(), Diffuse(),
            Specular(), ConstantAttenuation(), LinearAttenuation(),
            QuadraticAttenuation(), SpotDirection(), SpotExponent(),
            SpotCutoff(), Name(), LightOn());

         liList.push_back(newInfo);
      }
      selectedLight.pop();
   }
}

void LightManager::populateLightNameList(std::list<std::string> &lnList)
{
   std::list<LightBundle*>::iterator i;
   for (i = mLights.begin(); i != mLights.end(); i++)
   {
      selectedLight.push(*i);
      lnList.push_back(Name());
      selectedLight.pop();
   }
}

///// LIGHT PROPERTIES /////
void LightManager::LightNum(const int num)
{
   if (!isLightSelected( true ))   
      return;

   LightBundle * light = selectedLight.top();
   light->light->setLightNum(num);
   light->num = num;
}

int LightManager::LightNum()
{
   if (!isLightSelected( true ))
      return -1;

   return selectedLight.top()->light->getLightNum();
}

void LightManager::Ambient(const osg::Vec4 v4)
{
   if (!isLightSelected( true ))   
      return;   

   LightBundle * light = selectedLight.top();
   light->light->setAmbient(v4);
   light->modelMaterial->setAmbient(osg::Material::FRONT, v4+Diffuse());

   lightsChanged = true;
}

osg::Vec4 LightManager::Ambient()
{
   if (!isLightSelected( true ))  
      return osg::Vec4(0,0,0,0);

   return selectedLight.top()->light->getAmbient();
}

void LightManager::Diffuse(const osg::Vec4 v4)
{
   if (!isLightSelected( true ))
      return;

   LightBundle * light = selectedLight.top();
   light->light->setDiffuse(v4);
   light->modelMaterial->setAmbient(osg::Material::FRONT, v4+Ambient());

   lightsChanged = true;
}

osg::Vec4 LightManager::Diffuse()
{
   if (!isLightSelected( true ))   
      return osg::Vec4(-1.0,-1.0,-1.0,-1.0);
   
   return selectedLight.top()->light->getDiffuse();
}

void LightManager::Specular(const osg::Vec4 v4)
{
   if (!isLightSelected( true ))
      return;

   selectedLight.top()->light->setSpecular(v4);

   lightsChanged = true;
}

osg::Vec4 LightManager::Specular()
{
   if (!isLightSelected( true ))
      return osg::Vec4(-1.0,-1.0,-1.0,-1.0);
   
   return selectedLight.top()->light->getSpecular();
}

void LightManager::Position(const osg::Vec4 v4)
{
   if (!isLightSelected( true ))  
      return;

   LightBundle * light = selectedLight.top();

   // changing to or from directional light type?
   bool updateShape = light->light->getPosition().w() != v4.w();

   light->light->setPosition(v4);

   // new light type, so update the graphic
   if (updateShape)
      changeGraphicShape();

   updateGraphicTrans();

   lightsChanged = true;
}

osg::Vec4 LightManager::Position()
{
   if (!isLightSelected( true ))
      return osg::Vec4(0.0,0.0,0.0,-1.0);

   LightBundle * light = selectedLight.top();
   return light->light->getPosition();
}

void LightManager::SpotDirection(const osg::Vec3 v3)
{
   if (!isLightSelected( true ))  
      return;

   osg::Vec3d v3d = v3;
   v3d.normalize();

   LightBundle * light = selectedLight.top();

   light->light->setDirection(v3d);

   if (LightType() == SPOT)
      updateGraphicTrans();

   lightsChanged = true;
}

osg::Vec3 LightManager::SpotDirection()
{
   if (!isLightSelected( true ))
      return osg::Vec3(0.0, 0.0, 0.0);

   return selectedLight.top()->light->getDirection();
}

void LightManager::SpotExponent(const float f)
{
   if (!isLightSelected( true ))
   	return;

   selectedLight.top()->light->setSpotExponent(f);

   lightsChanged = true;
}

float LightManager::SpotExponent()
{
   if (!isLightSelected( true ))
   	return -1;

   return selectedLight.top()->light->getSpotExponent();
}

void LightManager::SpotCutoff(const float f)
{
   if (!isLightSelected( true ))
   	return;

   bool wasSpot = LightType() == SPOT;

   LightBundle * light = selectedLight.top();

   light->light->setSpotCutoff(f);

   if (f != 180.0)
   {
      light->lastValidCutoff = f;
   }

   if (wasSpot != (LightType() == SPOT))
      updateGraphicTrans();

   changeGraphicShape();

   lightsChanged = true;
}

float LightManager::SpotCutoff()
{
   if (!isLightSelected( true ))
   	return -1;
   return selectedLight.top()->light->getSpotCutoff();
}

void LightManager::ConstantAttenuation(const float f)
{
   if (!isLightSelected( true ))
   	return;

   selectedLight.top()->light->setConstantAttenuation(f);

   lightsChanged = true;
}

float LightManager::ConstantAttenuation()
{
   if (!isLightSelected( true ))
   	return -1;

   return selectedLight.top()->light->getConstantAttenuation();
}

void LightManager::LinearAttenuation(const float f)
{
   if (!isLightSelected( true ))
   	return;

   selectedLight.top()->light->setLinearAttenuation(f);

   lightsChanged = true;
}

float LightManager::LinearAttenuation()
{
   if (!isLightSelected( true ))
   	return -1;

   return selectedLight.top()->light->getLinearAttenuation();
}

void LightManager::QuadraticAttenuation(const float f)
{
   if (!isLightSelected( true ))
   	return;

   selectedLight.top()->light->setQuadraticAttenuation(f);

   lightsChanged = true;
}

float LightManager::QuadraticAttenuation()
{
   if (!isLightSelected( true ))
   	return -1;

   return selectedLight.top()->light->getQuadraticAttenuation();
}
///// END LIGHT PROPERTIES /////

void LightManager::PhysicalPosition(const osg::Vec4 v4)
{
   osg::Vec4 pos = Position();
   
   if (pos.w() == 1)
      Position(v4);
   else // pos.w() == 0
      Position(osg::Vec4(v4.x(),v4.y(),v4.z(),0));
}

osg::Vec4 LightManager::PhysicalPosition()
{
   osg::Vec4 pos = Position();
   return osg::Vec4(pos.x(),pos.y(),pos.z(),1);
}

void LightManager::LightType(const Type newType)
{
   // Validate Call
   if (!isLightSelected( true ))
      return;
      
   Type oldType = LightType();
   
   if (newType == oldType)
      return;
   
   // Change Light Properties
   if (newType == DIRECTIONAL)
   {
      osg::Vec4 pos = Position();
      Position(osg::Vec4(pos.x(),pos.y(),pos.z(),0));
   }
   else if (oldType == DIRECTIONAL)
   {
      osg::Vec4 pos = Position();
      Position(osg::Vec4(pos.x(),pos.y(),pos.z(),1));
   }

   LightBundle * light = selectedLight.top();
   
   if (newType == SPOT)
   {
      SpotCutoff(light->lastValidCutoff);
   }
   else if (oldType == SPOT)
   {
      SpotCutoff(180.0);
   }
}

LightManager::Type LightManager::LightType()
{
   if (!isLightSelected( true ))
      return POINT; // Need to return something.
   if (Position().w() == 0)
      return DIRECTIONAL;
   if (SpotCutoff() == 180.0)
      return POINT;
   return SPOT;
}

bool LightManager::LightOn()
{
   if (!isLightSelected( true ))
      return false;
      
   return selectedLight.top()->on;
}

///// LIGHT BUNDLE PROPERTIES /////
std::string LightManager::Name()
{
   if (!isLightSelected( true ))
      return "";

   return selectedLight.top()->name;
}
void LightManager::Name(const std::string n)
{
   if (!isLightSelected( true ))
      return;

   // Check for duplicate name
   
   std::list<LightBundle*>::iterator i;
   for (i = mLights.begin(); i != mLights.end(); i++)
   {
      selectedLight.push(*i);
      std::string name = Name();
      if (!name.compare(n))
      {
         std::cerr<<"Warning: Light named \""<<n<<"\" already exists. Leaving \""<<name<<"\" as is.\n";
         selectedLight.pop();
         return;
      }
      selectedLight.pop();
   }

   selectedLight.top()->name.assign(n);
}
void LightManager::Name(const char * n)
{
   if (!isLightSelected( true ))
      return;

   selectedLight.top()->name.assign(n);
}
///// END LIGHT BUNDLE PROPERTIES /////
void LightManager::GraphicSize(float size)
{
   graphicSize = size;
}

float LightManager::GraphicSize()
{
   return graphicSize;
}
