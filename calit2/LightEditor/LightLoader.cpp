#include "LightLoader.h"
#include <sstream>

void LightLoader::loadLights(const char * xmlFile, LightManager * manager)
{
   FILE *fp;

   fp = fopen(xmlFile, "r");

   if (fp == NULL) // Failed to open file.
   {
      std::cerr << "Error in LightLoader::loadLights - Could not open \"" << xmlFile << "\". Aborting.\n";
      return;
   }

   mxml_node_t * xmlTree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);

   // Add all the lights found in the tree
   std::cerr <<"LightEditor: Loading lights from "<<xmlFile<<" ... ";
   populateLightManager(xmlTree,manager);
   std::cerr <<"Done."<<std::endl;

   fclose(fp);
}

void LightLoader::saveLights(const char * xmlFile, LightManager * manager)
{
   FILE *fp;

   fp = fopen(xmlFile, "w");

   if (fp == NULL) // Failed to open file.
   {
      std::cerr << "Error in LightLoader::saveLights - Could not open \"" << xmlFile << "\". Aborting.\n";
      return;
   }

   mxml_node_t * xmlTree = mxmlNewXML("1.0");

   populateXmlTree(xmlTree, manager);

   std::cerr <<"LightEditor: Saving lights to "<<xmlFile<<" ... ";

   if (mxmlSaveFile(xmlTree, fp, LightLoader::format_cb) == -1)
      std::cerr<<" failed. Aborting.\n";
   else
      std::cerr<<"Done.\n";

   mxmlDelete(xmlTree);
   fclose(fp);
}

void LightLoader::populateLightManager(mxml_node_t * xmlTree, LightManager * manager)
{
   // Iterate through the xml tree, creating lights as you go
   mxml_node_t * editor = mxmlFindElement(xmlTree,xmlTree,"LightEditor",NULL,NULL,MXML_DESCEND_FIRST);

   if (editor == NULL)
   {
      std::cerr<<"\nWarning: No LightEditor tag in xml lights file. Aborting light loader... ";
      return;
   }
   // In the future, can do version control here, as necessary
   
   mxml_node_t * light = mxmlFindElement(editor,editor,"Light",NULL,NULL,MXML_DESCEND_FIRST);

   if (light)
   {
      // don't want to lose a reference to the current stored light (incase this was run mid-execution)
      manager->storeSelectedLight();

      do
      {
         // skip whitespace characters... odd that this is needed
         // seems very weird to use this.. no better way of checking?
         if (light->value.text.whitespace == 1)
            continue;

         manager->createNewLight();

         const char * name = mxmlElementGetAttr(light,"name");

         if (name != NULL)
            manager->Name(name);

         mxml_node_t * property = mxmlWalkNext(light,light,MXML_DESCEND);

         if (property == NULL) // default light settings
            continue;

         do
         {
            // skip whitespace characters... odd that this is needed
            // seems very weird to use this.. no better way of checking?
            if (property->value.text.whitespace == 1)
               continue;

            std::string propertyName = property->value.element.name;

            if (!propertyName.compare("Position"))
            {
               manager->Position(osg::Vec4(
                  floatFromProperty(property,"x"),
                  floatFromProperty(property,"y"),
                  floatFromProperty(property,"z"),
                  floatFromProperty(property,"w")
               ));
            }
            else if (!propertyName.compare("Ambient"))
            {
               manager->Ambient(osg::Vec4(
                  floatFromProperty(property,"r"),
                  floatFromProperty(property,"g"),
                  floatFromProperty(property,"b"),
                  floatFromProperty(property,"a")
               ));
            }
            else if (!propertyName.compare("Diffuse"))
            {
               manager->Diffuse(osg::Vec4(
                  floatFromProperty(property,"r"),
                  floatFromProperty(property,"g"),
                  floatFromProperty(property,"b"),
                  floatFromProperty(property,"a")
               ));
            }
            else if (!propertyName.compare("Specular"))
            {
               manager->Specular(osg::Vec4(
                  floatFromProperty(property,"r"),
                  floatFromProperty(property,"g"),
                  floatFromProperty(property,"b"),
                  floatFromProperty(property,"a")
               ));
            }
            else if (!propertyName.compare("ConstantAttenuation"))
            {
               manager->ConstantAttenuation(floatFromProperty(property,"float"));
            }
            else if (!propertyName.compare("LinearAttenuation"))
            {
               manager->LinearAttenuation(floatFromProperty(property,"float"));
            }
            else if (!propertyName.compare("QuadraticAttenuation"))
            {
               manager->QuadraticAttenuation(floatFromProperty(property,"float"));
            }
            else if (!propertyName.compare("SpotDirection"))
            {
               manager->SpotDirection(osg::Vec3(
                  floatFromProperty(property,"x"),
                  floatFromProperty(property,"y"),
                  floatFromProperty(property,"z")
               ));
            }
            else if (!propertyName.compare("SpotExponent"))
            {
               manager->SpotExponent(floatFromProperty(property,"float"));
            }
            else if (!propertyName.compare("SpotCutoff"))
            {
               manager->SpotCutoff(floatFromProperty(property,"float"));
            }
            else
            {
               std::cerr<<"\nProperty \""<<propertyName<<"\" in light xml file under \""<<name<<"\". Ignoring property.";
            }

         } while ((property = mxmlWalkNext(property,light,MXML_NO_DESCEND)) != NULL);
      } while ((light = mxmlWalkNext(light,editor,MXML_NO_DESCEND)) != NULL);

      // restore the stored light
      manager->recallStoredLight();
   }
}

float LightLoader::floatFromProperty(mxml_node_t * property, char * attribute)
{
   float f;

   std::stringstream ss;

   ss << mxmlElementGetAttr(property,attribute);
   ss >> f;

   return f;
}

void LightLoader::populateXmlTree(mxml_node_t * xmlTree, LightManager * manager)
{
   mxml_node_t * data = mxmlNewElement(xmlTree,"LightEditor");

   const int STR_SIZE = 50;
   char attrValue[STR_SIZE];

   mxmlElementSetAttrf(data, "version", "%d", LightManager::VERSION);

   // Get light information
   std::list<LightManager::LightInfo*> infoList;
   manager->populateLightInfoList(infoList);

   // Populate tree given the lights in manager. Use populateLightInfoList.
   mxml_node_t * light, * property;

   std::list<LightManager::LightInfo*>::iterator i;
   for (i = infoList.begin(); i != infoList.end(); i++)
   {
      // Group Light by Name
      light = mxmlNewElement(data, "Light");
      mxmlElementSetAttr(light,"name",(*i)->name.c_str());
      // Position
      property = mxmlNewElement(light,"Position");
      mxmlElementSetAttrf(property,"x","%f",(*i)->position.x());
      mxmlElementSetAttrf(property,"y","%f",(*i)->position.y());
      mxmlElementSetAttrf(property,"z","%f",(*i)->position.z());
      mxmlElementSetAttrf(property,"w","%f",(*i)->position.w());
      // Ambient
      property = mxmlNewElement(light,"Ambient");
      mxmlElementSetAttrf(property,"r","%f",(*i)->ambient.x());
      mxmlElementSetAttrf(property,"g","%f",(*i)->ambient.y());
      mxmlElementSetAttrf(property,"b","%f",(*i)->ambient.z());
      mxmlElementSetAttrf(property,"a","%f",(*i)->ambient.w());
      // Diffuse
      property = mxmlNewElement(light,"Diffuse");
      mxmlElementSetAttrf(property,"r","%f",(*i)->diffuse.x());
      mxmlElementSetAttrf(property,"g","%f",(*i)->diffuse.y());
      mxmlElementSetAttrf(property,"b","%f",(*i)->diffuse.z());
      mxmlElementSetAttrf(property,"a","%f",(*i)->diffuse.w());
      // Specular
      property = mxmlNewElement(light,"Specular");
      mxmlElementSetAttrf(property,"r","%f",(*i)->specular.x());
      mxmlElementSetAttrf(property,"g","%f",(*i)->specular.y());
      mxmlElementSetAttrf(property,"b","%f",(*i)->specular.z());
      mxmlElementSetAttrf(property,"a","%f",(*i)->specular.w());
      // ConstantAttenuation
      property = mxmlNewElement(light,"ConstantAttenuation");
      snprintf(attrValue,STR_SIZE,"%f",(*i)->constant);
      mxmlElementSetAttr(property,"float",attrValue);
      // LinearAttenuation
      property = mxmlNewElement(light,"LinearAttenuation");
      snprintf(attrValue,STR_SIZE,"%f",(*i)->linear);
      mxmlElementSetAttr(property,"float",attrValue);
      // QuadraticAttenuation
      property = mxmlNewElement(light,"QuadraticAttenuation");
      snprintf(attrValue,STR_SIZE,"%f",(*i)->quadratic);
      mxmlElementSetAttr(property,"float",attrValue);
      // SpotDirection
      property = mxmlNewElement(light,"SpotDirection");
      snprintf(attrValue,STR_SIZE,"%f",(*i)->spotDirection.x());
      mxmlElementSetAttr(property,"x",attrValue);
      snprintf(attrValue,STR_SIZE,"%f",(*i)->spotDirection.y());
      mxmlElementSetAttr(property,"y",attrValue);
      snprintf(attrValue,STR_SIZE,"%f",(*i)->spotDirection.z());
      mxmlElementSetAttr(property,"z",attrValue);
      // SpotExponent
      property = mxmlNewElement(light,"SpotExponent");
      snprintf(attrValue,STR_SIZE,"%f",(*i)->spotExponent);
      mxmlElementSetAttr(property,"float",attrValue);
      // SpotCutoff
      property = mxmlNewElement(light,"SpotCutoff");
      snprintf(attrValue,STR_SIZE,"%f",(*i)->spotCutoff);
      mxmlElementSetAttr(property,"float",attrValue);
   }
}

const char * LightLoader::format_cb(mxml_node_t *node, int where)
{
   std::string name = node->value.element.name;

   if (where == MXML_WS_AFTER_OPEN || where == MXML_WS_AFTER_CLOSE)
      return "\n";

   if (name[0] == '?' || !name.compare("LightEditor")) // opening flags
   {
      return NULL; // no spacing before hand
   }

   if (!name.compare("Light"))
      return "\t";

   return "\t\t";
}
