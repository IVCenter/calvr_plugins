/** Light Manager Class used to handle the different lights created
    during the lifetime of the plugin
*/

#ifndef _LIGHT_LOADER_H
#define _LIGHT_LOADER_H

// Std 
#include <iostream>

// Mini-XML
#include <mxml.h>

#include "LightManager.h"

class LightLoader
{
   public:
      // XML File Operations
      static void loadLights(const char * xmlFile, LightManager * manager); 
      static void saveLights(const char * xmlFile, LightManager * manager); 
      // End XML File Operations
   private:
      LightLoader();
      ~LightLoader();

      static void populateLightManager(mxml_node_t * xmlTree, LightManager * manager);
      static void populateXmlTree(mxml_node_t * xmlTree, LightManager * manager);
      static float floatFromProperty(mxml_node_t * property, char * attribute);
      static const char * format_cb(mxml_node_t *node, int where);
};

#endif
