#include "OsgVnc.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/NodeMask.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrMenu/MenuSystem.h>
#include <PluginMessageType.h>
#include <iostream>

#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Texture2D>
#include <osg/TextureRectangle>
#include <osg/TextureCubeMap>
#include <osg/TexMat>
#include <osg/CullFace>
#include <osg/ImageStream>
#include <osg/io_utils>
#include <osgDB/Registry>

#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

using namespace osg;
using namespace std;
using namespace cvr;

// browser query commands
const int buffsize = 2048;
const std::string baseQuery = "window.location='http://www.google.com/search?as_q=";

CVRPLUGIN(OsgVnc)

OsgVnc::OsgVnc() : FileLoadCallback("vnc")
{
}

bool OsgVnc::loadFile(std::string filename)
{
	// set defaultScale and position
    float scale = _defaultScale;
    osg::Vec3f pos;

    osgWidget::GeometryHints hints(osg::Vec3(0.0f,0.0f,0.0f),
                                   osg::Vec3(1.0f,0.0f,0.0f),
                                   osg::Vec3(0.0f,0.0f,1.0f),
                                   osg::Vec4(1.0f,1.0f,1.0f,1.0f),
                                   osgWidget::GeometryHints::RESIZE_HEIGHT_TO_MAINTAINCE_ASPECT_RATIO);

	// check if there is a configuration for the file
    std::map<std::string, std::pair<float, osg::Vec3f> >::iterator it = _locInit.find(filename);
    if( it != _locInit.end() )
    {
        scale = it->second.first;
        pos.set(it->second.second);
    }
   
    std::string hostname = osgDB::getNameLessExtension(filename);
    osg::ref_ptr<osgWidget::VncClient> vncClient = new osgWidget::VncClient;
    if (vncClient->connect(hostname, hints))
    {
        struct VncObject * currentobject = new struct VncObject;
        currentobject->name = hostname;

        // add to scene object
        VncSceneObject * sot = new VncSceneObject(currentobject->name, vncClient.get(), ComController::instance()->isMaster(), false,false,false,true,true);
        PluginHelper::registerSceneObject(sot,"OsgVnc");

		// set up SceneObject
        sot->setNavigationOn(false);
        sot->setMovable(true);
        sot->addMoveMenuItem();
        // will allow to resize to a 10th of a default size and 10 times default size
        sot->addScaleMenuItem("Scale", scale * 0.1, scale * 10.0, scale);
        sot->setPosition(pos);
        sot->setScale(scale);
        sot->attachToScene();

        currentobject->scene = sot;
                
        MenuButton * mb = new MenuButton("Delete");
        mb->setCallback(this);
        sot->addMenuItem(mb);
        _deleteMap[currentobject] = mb;

        _loadedVncs.push_back(currentobject);
    }
    else
    {
	    std::cerr << "Unable to read vnc stream\n";
    }

    return true;
}

void OsgVnc::menuCallback(MenuItem* menuItem)
{

    //check map for a delete
    for(std::map<struct VncObject*, MenuButton*>::iterator it = _deleteMap.begin(); it != _deleteMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            for(std::vector<struct VncObject*>::iterator delit = _loadedVncs.begin(); delit != _loadedVncs.end(); delit++)
            {
                if((*delit) == it->first)
                {
		            // need to delete title SceneObject
		            if( it->first->scene )
			            delete it->first->scene;
                    it->first->scene = NULL; 
    
                    _loadedVncs.erase(delit);
                    break;
                }
            }

            delete it->first;
            delete it->second;
            _deleteMap.erase(it);

            break;
        }
    }

 	//check for main menu selections
    std::map< cvr::MenuItem* , std::string>::iterator it = _menuFileMap.find(menuItem);
    if( it != _menuFileMap.end() )
    {
        loadFile(it->second);
        return;
    }

    // check for remove all button
    if( menuItem == _removeButton )
    {
        removeAll();
    }

    if( menuItem == _hideCheckbox )
    {
        hideAll(_hideCheckbox->getValue());
    }

}

bool OsgVnc::init()
{
    std::cerr << "OsgVnc init\n";

	// create default menu
    _vncMenu = new SubMenu("OsgVnc", "OsgVnc");
    _vncMenu->setCallback(this);

    _sessionsMenu = new SubMenu("Sessions","Sessions");
    _sessionsMenu->setCallback(this);
    _vncMenu->addItem(_sessionsMenu);

    _hideCheckbox = new MenuCheckbox("Hide",false);
    _hideCheckbox->setCallback(this);
    _vncMenu->addItem(_hideCheckbox);

    _removeButton = new MenuButton("Remove All");
    _removeButton->setCallback(this);
    _vncMenu->addItem(_removeButton);

    // read in default values
    _defaultScale = ConfigManager::getFloat("Plugin.OsgVnc.DefaultScale", 2048.0);

    // read in configurations
    _configPath = ConfigManager::getEntry("Plugin.OsgVnc.ConfigDir");

    ifstream cfile;
    cfile.open((_configPath + "/Init.cfg").c_str(), ios::in);

	if(!cfile.fail())
    {
        string line;
        while(!cfile.eof())
        {
            osg::Vec3 p;
            float scale;
            char name[150];
            cfile >> name;
            if(cfile.eof())
            {
                break;
            }
            cfile >> scale;
            for(int i = 0; i < 3; i++)
            {
                    cfile >> p[i];
            }
            _locInit[string(name)] = pair<float, osg::Vec3>(scale, p);
        }
    }
    cfile.close();

    // read in configuartion files
    vector<string> list;

    string configBase = "Plugin.OsgVnc.Sessions";

    ConfigManager::getChildren(configBase,list);

	for(int i = 0; i < list.size(); i++)
    {
        MenuButton * button = new MenuButton(list[i]);
        button->setCallback(this);

        // add mapping
        _menuFileMap[button] = ConfigManager::getEntry("value",configBase + "." + list[i],"");

        // add button
        _sessionsMenu->addItem(button);
    }

    // add menu
    cvr:MenuSystem::instance()->addMenuItem(_vncMenu);

    return true;
}

void OsgVnc::writeConfigFile()
{
    ofstream cfile;
    cfile.open((_configPath + "/Init.cfg").c_str(), ios::trunc);

    if(!cfile.fail())
    {
        for(map<std::string, std::pair<float, osg::Vec3> >::iterator it = _locInit.begin();
                it != _locInit.end(); it++)
        {
            cfile << it->first << " " << it->second.first << " ";
            for(int i = 0; i < 3; i++)
            {
                cfile << it->second.second[i] << " ";
            }
            cfile << endl;
        }
    }
    cfile.close();
}


void OsgVnc::message(int type, char *&data, bool collaborative)
{
	if(type == VNC_GOOGLE_QUERY)
    {
        if(collaborative)
        {
            return;
        }

        std::string hostname = ConfigManager::getEntry("Plugin.OsgVnc.BrowserQueryServer");
        int port = ConfigManager::getInt("Plugin.OsgVnc.Port", 32000);

        // try and send request to remote firefox browser running mozrepl plugin
        OsgVncRequest* request = (OsgVncRequest*)data;
        launchQuery(hostname, port, request->query);
    }

    if( type == VNC_HIDE )
    {
        if( _loadedVncs.size() > 0 )
        {
            OsgVncRequest* request = (OsgVncRequest*)data;
            hideAll(request->hide);
        }
    }

    if( type == VNC_SCALE )
    {
        if( _loadedVncs.size() > 0 )
        {
            OsgVncRequest* request = (OsgVncRequest*)data;
            _loadedVncs.at(0)->scene->setScale(request->scale);
        }
    }

    if( type == VNC_POSITION )
    {
        if( _loadedVncs.size() > 0 )
        {
            OsgVncRequest* request = (OsgVncRequest*)data;
            _loadedVncs.at(0)->scene->setPosition(request->position);
        }
    }
}

void OsgVnc::launchQuery(std::string& hostname, int portno, std::string& query)
{
	int sockfd, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    if (sockfd < 0)
	{
        std::cerr << "ERROR opening socket\n";
		return;	
	}

    server = gethostbyname(hostname.c_str());

    if (server == NULL) 
	{
        std::cerr << "ERROR, no such host\n";
        return;
    }

    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr,
          (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);
    serv_addr.sin_port = htons(portno);

    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
	{
        std::cerr << "ERROR connecting\n";
		return;
	}

    std::stringstream ss;
    ss << baseQuery;
    ss << query;
    ss << "'";

    // write initial debug message
    n = write(sockfd, ss.str().c_str(), ss.str().size());
    if (n < 0)
	{
        std::cerr << "ERROR writing to socket\n";
		return;
	}

	// read response
    char buffer[buffsize];
    bzero(buffer,buffsize);
    n = read(sockfd,buffer,buffsize);

    if(n < 0)
    {
       std::cerr << "ERROR reading from socket\n";
       return;
    }
    close(sockfd);
}

void OsgVnc::hideAll(bool hide)
{
    for(int i = 0; i < _loadedVncs.size(); i++)
    {
         VncObject* it = _loadedVncs.at(i);

         if( hide )
         {
            PluginHelper::unregisterSceneObject(it->scene);
         }
         else
         {
            PluginHelper::registerSceneObject(it->scene,"OsgVnc");
            it->scene->attachToScene();
         }
    }
}


void OsgVnc::removeAll()
{
    while(_loadedVncs.size())
    {
         VncObject* it = _loadedVncs.at(0);

         // remove delete map item
         if(_deleteMap.find(it) != _deleteMap.end())
         {
            delete _deleteMap[it];
            _deleteMap.erase(it);
         }

         if( it->scene )
                delete it->scene;
         it->scene = NULL;

         _loadedVncs.erase(_loadedVncs.begin());
    }
}


OsgVnc::~OsgVnc()
{
    removeAll();
}
