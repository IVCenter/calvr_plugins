// **************************************************************************
//
// Description:   StructView
//
// Author:        Andre Barbosa
//
// Creation Date: 2007-01-26
//
// **************************************************************************

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <string.h>

// OSG:
#include <osg/Node>
#include <osg/Group>
#include <osg/Switch>
#include <osgDB/ReadFile>

// Local:
#include "StructView.h"

using namespace std;
using namespace osg;
using namespace cvr;

CVRPlugin* StructView::plugin = NULL;

// Constructor
StructView::StructView()
{
}

bool StructView::init()
{
  if(plugin != NULL)
  {
    return false;
  }
  plugin = this;

  mainNode = new Switch;
  mainNode->setName("StructView");
 
  //Create main menu button
  structViewMenu = new SubMenu("StructView","StructView");
  structViewMenu->setCallback(this);
  PluginHelper::addRootMenuItem(structViewMenu);
  
  // Create StructView menu
  enable = new MenuCheckbox("Enable", false);
  MenuCheckbox * cmi;
  cmi = new MenuCheckbox("Structural Steel 01", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Structural Steel 02", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Rebar Top", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Rebar Bottom", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Rebar CutOut", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Rebar Walls", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Rebar Fenders", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Rebar Skirt T", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Rebar Skirt BW", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Drainage", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Concrete", false);
  menuList.push_back(cmi);
  cmi = new MenuCheckbox("Tower", false);
  menuList.push_back(cmi);
  enable->setCallback(this);
  structViewMenu->addItem(enable);

  for(vector<MenuCheckbox*>::iterator it = menuList.begin(); it != menuList.end(); it++)
  {
    (*it)->setCallback(this);
    structViewMenu->addItem(*it);
  }

  return true;
}

/// Destructor
StructView::~StructView()
{
  PluginHelper::getObjectsRoot()->removeChild(mainNode.get());
  delete structViewMenu;
  delete enable;
  for(vector<MenuCheckbox*>::iterator it = menuList.begin(); it != menuList.end(); it++)
  {
    delete *it;
  }
}
  
void StructView::menuCallback(MenuItem* menuItem)
{
  if(menuItem == enable)
  {
    //cerr << "Enable: " << enable->getValue() << endl;
    if(enable->getValue())
    {
      PluginHelper::getObjectsRoot()->addChild(mainNode.get());
      // set default directory and layer file names
      std::string layerFile;
      for(int i=0;i<12;i++)
      {
        char entry[50];
        sprintf(entry,"Plugin.StructView.Layer%dFile",i);
        layerFile = ConfigManager::getEntry(entry);
        if (!layerFile.empty())
        {
          Node* layerNode = osgDB::readNodeFile(layerFile);
          if(layerNode)
          {
            layerNode->setNodeMask(0);
            mainNode->addChild(layerNode);
          }
          else
          {
            cerr << "reading " << layerFile << " failed" << endl;
          }
        }
      }
    }
    else
    {
      mainNode->removeChildren(0, mainNode->getNumChildren());
      PluginHelper::getObjectsRoot()->removeChild(mainNode.get());
      for(vector<MenuCheckbox*>::iterator it = menuList.begin(); it != menuList.end(); it++)
      {
        (*it)->setValue(false);
      }
    }
    return;
  }

  for(int i = 0; i < menuList.size(); i++)
  {
    if(menuItem == menuList[i])
    {
      if(enable->getValue())
      {
        if(menuList[i]->getValue())
        {
          mainNode->getChild(i)->setNodeMask(~0);
        }
        else
        {
          mainNode->getChild(i)->setNodeMask(0);
        }
      }
      else
      {    
        menuList[i]->setValue(false);
      }
      return;
    }
  }
}

CVRPLUGIN(StructView)
