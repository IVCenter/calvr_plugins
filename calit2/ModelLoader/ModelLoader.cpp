#include "ModelLoader.h"

#include <config/ConfigManager.h>
#include <kernel/SceneManager.h>
#include <menu/MenuSystem.h>
#include <iostream>

#include <osg/Matrix>
#include <osgDB/ReadFile>

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN(ModelLoader)

ModelLoader::ModelLoader() : FileLoadCallback("vrml,obj")
{

}

bool ModelLoader::init()
{
    std::cerr << "ModelLoader init\n";

    wasInit = 0;

    root = new osg::MatrixTransform();

    MLMenu = new SubMenu("ModelLoader", "ModelLoader");
    MLMenu->setCallback(this);

    loadMenu = new SubMenu("Load","Load");
    loadMenu->setCallback(this);
    MLMenu->addItem(loadMenu);

    loadButton = new MenuButton("Load Position");
    loadButton->setCallback(this);
    MLMenu->addItem(loadButton);

    saveMenu = new SubMenu("Save Position", "Save Position");
    saveMenu->setCallback(this);
    MLMenu->addItem(saveMenu);

    saveButton = new MenuButton("Save");
    saveButton->setCallback(this);
    saveMenu->addItem(saveButton);

    removeButton = new MenuButton("Remove");
    removeButton->setCallback(this);
    MLMenu->addItem(removeButton);

    vector<string> list;

    string configBase = "Plugin.ModelLoader.Files";

    ConfigManager::getChildren(configBase,list);

    for(int i = 0; i < list.size(); i++)
    {
	MenuButton * button = new MenuButton(list[i]);
	button->setCallback(this);
	menuFileList.push_back(button);

	struct loadinfo * info = new struct loadinfo;
	info->name = list[i];
	info->path = ConfigManager::getEntry("path",configBase + "." + list[i],"");
	info->mask = ConfigManager::getInt("mask",configBase + "." + list[i], 1);
	info->lights = ConfigManager::getInt("lights",configBase + "." + list[i], 1);

	models.push_back(info);
    }

    configPath = ConfigManager::getEntry("Plugin.ModelLoader.ConfigDir");

    ifstream cfile;
    cfile.open((configPath + "/Init.cfg").c_str(), ios::in);

    if(!cfile.fail())
    {
	string line;
	while(!cfile.eof())
	{
	    Matrix m;
	    float scale;
	    char name[150];
	    cfile >> name;
	    if(cfile.eof())
	    {
		break;
	    }
	    cfile >> scale;
	    for(int i = 0; i < 4; i++)
	    {
		for(int j = 0; j < 4; j++)
		{
		    cfile >> m(i, j);
		}
	    } 
	    locInit[string(name)] = pair<float, Matrix>(scale, m);
	}
    }
    cfile.close();

    for(int k = 0; k < menuFileList.size(); k++)
    {
	loadMenu->addItem(menuFileList[k]);
    }

    MenuSystem::instance()->addMenuItem(MLMenu);

    SceneManager::instance()->getObjectsRoot()->addChild(root);

    std::cerr << "ModelLoader init done.\n";
    return true;
}


ModelLoader::~ModelLoader()
{
}

void ModelLoader::menuCallback(MenuItem* menuItem)
{
    if(menuItem == removeButton)
    {
        if(root->getNumChildren() != 0)
        {
            root->removeChildren(0, root->getNumChildren());
        }
        wasInit = 0;
        return;
    }

    if(menuItem == loadButton)
    {
        if(!wasInit)
        {
            return;
        }

        if(locInit.find(models[loadedModel]->name) != locInit.end())
        {
	    SceneManager::instance()->setObjectScale(locInit[models[loadedModel]->name].first);
	    SceneManager::instance()->setObjectMatrix(locInit[models[loadedModel]->name].second);
        }

        return;
    }

    if(menuItem == saveButton)
    {
        if(!wasInit)
        {
            return;
        }

        locInit[models[loadedModel]->name] = pair<float, Matrix>(SceneManager::instance()->getObjectScale(), SceneManager::instance()->getObjectTransform()->getMatrix());
        writeConfigFile();

        return;
    }

    for(int i = 0; i < menuFileList.size(); i++)
    {
        if(menuFileList[i] == menuItem)
        {
            if(root->getNumChildren() != 0)
            {
                root->removeChildren(0, root->getNumChildren());
            }

            osg::Node* modelNode = osgDB::readNodeFile(models[i]->path);
            if(modelNode==NULL)
            { 
                std::cerr << "ModelLoader: Error reading file " << models[i]->path << endl;
                return;
            }
            else
            {
                if(models[i]->mask)
                {
                    modelNode->setNodeMask(modelNode->getNodeMask() & ~2);
                }
                if(!models[i]->lights)
                {
                    osg::StateSet* stateset = modelNode->getOrCreateStateSet();
                    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
                }
            }

            root->addChild(modelNode);
            
            if(locInit.find(models[i]->name) != locInit.end())
            {
                SceneManager::instance()->setObjectScale(locInit[models[i]->name].first);
                SceneManager::instance()->setObjectMatrix(locInit[models[i]->name].second);
            }

            wasInit = 1;
            loadedModel = i;
        }
    }
}

bool ModelLoader::loadFile(std::string file)
{
    std::cerr << "ModelLoader: Loading file: " << file << std::endl;
    if(root->getNumChildren() != 0)
    {
        root->removeChildren(0, root->getNumChildren());
    }

    wasInit = 0;

    osg::Node* modelNode = osgDB::readNodeFile(file);
    if(modelNode==NULL)
    {
        cerr << "ModelLoader: Error reading file " << file << endl;
        return false;
    }
    else
    {
        modelNode->setNodeMask(modelNode->getNodeMask() & ~2);
    }

    root->addChild(modelNode);

    return true;
}

void ModelLoader::writeConfigFile()
{
    ofstream cfile;
    cfile.open((configPath + "/Init.cfg").c_str(), ios::trunc);

    if(!cfile.fail())
    {
       for(map<std::string, std::pair<float, osg::Matrix> >::iterator it = locInit.begin(); it != locInit.end(); it++)
       {
           //cerr << "Writing entry for " << it->first << endl;
           cfile << it->first << " " << it->second.first << " ";
           for(int i = 0; i < 4; i++)
           {
               for(int j = 0; j < 4; j++)
               {
                   cfile << it->second.second(i, j) << " ";
               }
           }
           cfile << endl;
       }
    }
    cfile.close();
}
