#include "TourCave.h"

#include <iostream>
#include <sstream>

#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/ComController.h>
#include <cvrConfig/ConfigManager.h>
#include <PluginMessageType.h>

using namespace std;

CVRPLUGIN(TourCave)

const char * PathPlugin = "OsgPathRecorder";

TourCave::TourCave()
{
    _currentMode = -1;
    _backgroundAudio = NULL;
    _mls = NULL;
    _mStatus = DONE;

    if(ComController::instance()->isMaster())
    {
#ifdef HAS_AUDIO
	std::string base = ConfigManager::getEntry("Plugin.TourCave.AudioBase");
	std::string server = ConfigManager::getEntry("Audio.AssetManager.Server");
	_am = new AssetManager(base, server);
	_am->Init();
	_am->SetVolume(0.0);
#endif
    }

}

TourCave::~TourCave()
{
    if(_mls)
    {
	delete _mls;
    }

    for(int i = 0; i < _socketList.size(); i++)
    {
	delete _socketList[i];
    }

    if(ComController::instance()->isMaster())
    {
#ifdef HAS_AUDIO
	_am->SendUDP("/Ambient/All/Play", "i", 0);
	_am->SendUDP("/Voice/All/Play", "i", 0);
	_am->Destroy();
	delete _am;
#endif
    }
}

bool TourCave::init()
{
    bool found;

    if(ComController::instance()->isMaster())
    {
	_mls = new MultiListenSocket(11011);
	if(!_mls->setup())
	{
	    std::cerr << "Error setting up MultiListen Socket." << std::endl;
	    delete _mls;
	    _mls = NULL;
	}
    }

    std::string name;

    name = ConfigManager::getEntry("name","Plugin.TourCave.BackgroundAudio","NONE",&found);

    if(found)
    {
	_backgroundAudio = new TourAudio;
	_backgroundAudio->name = name;
	_backgroundAudio->loop = ConfigManager::getBool("loop","Plugin.TourCave.BackgroundAudio",false, NULL);
        _backgroundAudio->volume = ConfigManager::getFloat("volume","Plugin.TourCave.BackgroundAudio",1.0);
	_backgroundAudio->started = false;

        if(ComController::instance()->isMaster())
	{
	    //TODO: start?, or wait for some sort of trigger
#ifdef HAS_AUDIO
	    //_am->SendUDP((name + "/Volume").c_str(),"f",1.0 * ta->volume);
	    _am->SendUDP((name + "/Play").c_str(),"i",1);
#endif
	}
    }

    _tourCaveMenu = new SubMenu("TourCave","TourCave");
    PluginHelper::addRootMenuItem(_tourCaveMenu);

    int path;
    int mode = 0;

    std::stringstream ss;
    ss << "Mode" << mode;

    path = ConfigManager::getInt(std::string("Plugin.TourCave.") + ss.str() + ".PathID", 0, &found);
    while(found)
    {
	std::stringstream ss1;
	ss1 << "Mode" << mode;

       float speed = ConfigManager::getFloat("speed",std::string("Plugin.TourCave.") + ss1.str() + ".PathID",1.0);
       _modePathSpeedList.push_back(speed);

	MenuButton * mb = new MenuButton(ss1.str());
	mb->setCallback(this);
	_tourCaveMenu->addItem(mb);
	_modeButtonList.push_back(mb);

	_modePathIDList.push_back(path);

	_modeAudioList.push_back(std::vector<TourAudio*>());
	std::vector<std::string> audioList;
	ConfigManager::getChildren(std::string("Plugin.TourCave.") + ss1.str() + ".Audio", audioList);

	for(int i = 0; i < audioList.size(); i++)
	{
	    std::stringstream entry;
	    entry << "Plugin.TourCave." << ss1.str() << ".Audio." << audioList[i];
	    TourAudio * ta;
	    ta = new TourAudio;
	    ta->name = ConfigManager::getEntry("name",entry.str(),"name");
	    ta->triggerTime = ConfigManager::getFloat("time",entry.str(),0);
	    ta->loop = ConfigManager::getBool("loop",entry.str(),false);
	    ta->started = false;

	    bool usedist;

	    float x,y,z,dist;
	    dist = ConfigManager::getFloat("distance",entry.str(),0,&usedist);
	    if(usedist)
	    {
		ta->distance = dist;
		x = ConfigManager::getFloat("x",entry.str());
		y = ConfigManager::getFloat("y",entry.str());
		z = ConfigManager::getFloat("z",entry.str());
		ta->pos = osg::Vec3(x,y,z);
	    }
	    else
	    {
		ta->distance = 0;
	    }

            std::string s;
	    s = ConfigManager::getEntry("stopLocation",entry.str(),"NEXT");
            if(s == "NEXT")
            {
                ta->sl = NEXT_PATH;
            }
            else
            {
                ta->sl = PATH_END;
            }

            s = ConfigManager::getEntry("stopAction",entry.str(),"STOP");
            if(s == "NOTHING")
            {
                ta->sa = NOTHING;
            }
            else if(s == "PAUSE")
            {
                ta->sa = PAUSE;
            }
            else
            {
                ta->sa = STOP;
            }

            ta->volume = ConfigManager::getFloat("volume",entry.str(),1.0);

	    _modeAudioList[mode].push_back(ta);
	}

	mode++;
	std::stringstream ss2;
	ss2 << "Mode" << mode;

	path = ConfigManager::getInt(std::string("Plugin.TourCave.") + ss2.str() + ".PathID", 0, &found);
    }

    return true;
}

void TourCave::preFrame()
{
    int numCommands;
    char * commands;
    if(ComController::instance()->isMaster())
    {
	if(_mls)
	{
	    CVRSocket * con;
	    if((con = _mls->accept()))
	    {
		std::cerr << "Adding socket." << std::endl;
		con->setNoDelay(true);
		_socketList.push_back(con);
	    }
	}

	checkSockets();

	numCommands = _commandList.size();

	ComController::instance()->sendSlaves(&numCommands, sizeof(int));

	if(numCommands)
	{
	    commands = new char[numCommands];
	    for(int i = 0; i < numCommands; i++)
	    {
		commands[i] = _commandList[i];
	    }
	    _commandList.clear();
	    ComController::instance()->sendSlaves(commands,numCommands * sizeof(char));
	}
    }
    else
    {
	ComController::instance()->readMaster(&numCommands, sizeof(int));
	if(numCommands)
	{
	    commands = new char[numCommands];
	    ComController::instance()->readMaster(commands,numCommands * sizeof(char));
	}
    }

    if(numCommands)
    {
	processCommands(commands, numCommands);
	delete[] commands;
    }

    if(_mStatus == STARTED)
    {
	PluginManager::instance()->sendMessageByName(PathPlugin,PR_IS_STOPPED,(char*)"TourCave");
    }

    if(_currentMode >= 0 && _mStatus == STARTED)
    {
	PluginManager::instance()->sendMessageByName(PathPlugin,PR_GET_TIME,(char *)"TourCave");

	for(int i = 0; i < _modeAudioList[_currentMode].size(); i++)
	{
	    TourAudio * ta = _modeAudioList[_currentMode][i];
	    if(!ta->started)
	    {
		if(_currentTime >= ta->triggerTime)
		{
		    // start audio
		    
		    std::cerr << "Starting audio file " << ta->name << std::endl;

		    if(ComController::instance()->isMaster())
		    {
#ifdef HAS_AUDIO
			if(!ta->distance)
			{
			    //_am->SendUDP((ta->name + "/Volume").c_str(),"f",1.0 * ta->volume);
			}
			else
			{
			    //_am->SendUDP((ta->name + "/Volume").c_str(),"f",0.0);
			}
			_am->SendUDP((ta->name + "/Play").c_str(),"i",1);
#endif
		    }

		    ta->started = true;
		}
	    }
	    else if(ta->distance)
	    {
		osg::Vec3 dist = (ta->pos * PluginHelper::getObjectToWorldTransform()) - PluginHelper::getHeadMat(0).getTrans();
		float f = dist.length();

		if(ComController::instance()->isMaster())
		{
#ifdef HAS_AUDIO
		    float rampup = 0.25;
		    if(ta->distance <= f)
		    {
			//_am->SendUDP((ta->name + "/Volume").c_str(),"f",1.0 * ta->volume);
		    }
		    else if((f - ta->distance) < ta->distance * rampup)
		    {
			float frac = (f - ta->distance) / (ta->distance * rampup);
			frac = 1.0 - frac;
			//_am->SendUDP((ta->name + "/Volume").c_str(),"f",frac * ta->volume); 
		    }

#endif
		}
	    }
	}
    }
}

void TourCave::menuCallback(MenuItem * item)
{

    for(int i = 0; i < _modeButtonList.size(); i++)
    {
	if(item == _modeButtonList[i])
	{
	    if(_currentMode >= 0)
	    {
		PluginManager::instance()->sendMessageByName(PathPlugin,PR_STOP,NULL);

		for(int j = 0; j < _modeAudioList[_currentMode].size(); j++)
		{
		    if(ComController::instance()->isMaster())
		    {
			// stop audio
#ifdef HAS_AUDIO
			if(_mStatus == STARTED)
			{
			    std::cerr << "Stopping audio file: " << _modeAudioList[_currentMode][j]->name << std::endl;
			    _am->SendUDP((_modeAudioList[_currentMode][j]->name + "/Stop").c_str(),"i", 1);
			}
			else
			{
			    if(_modeAudioList[_currentMode][j]->sl == NEXT_PATH)
			    {
				if(_modeAudioList[_currentMode][j]->sa == STOP)
				{
				    std::cerr << "Stopping audio file: " << _modeAudioList[_currentMode][j]->name << std::endl;
				    _am->SendUDP((_modeAudioList[_currentMode][j]->name + "/Stop").c_str(),"i", 1);
				}
				else if(_modeAudioList[_currentMode][j]->sa == PAUSE)
				{
				    std::cerr << "Pause audio file: " << _modeAudioList[_currentMode][j]->name << std::endl;
				    _am->SendUDP((_modeAudioList[_currentMode][j]->name + "/Pause").c_str(),"i", 1);
				}
			    }
			}
#endif
		    }
		}
	    }

	    _currentMode = i;

	    _mStatus = STARTED;

	    for(int j = 0; j < _modeAudioList[_currentMode].size(); j++)
	    {
		_modeAudioList[_currentMode][j]->started = false;
	    }

	    bool b = true;

            PluginManager::instance()->sendMessageByName(PathPlugin,PR_SET_ACTIVE_ID,(char *)&_modePathIDList[_currentMode]);
	    PluginManager::instance()->sendMessageByName(PathPlugin,PR_SET_PLAYBACK,(char *)&b);
            std::cerr << "Sending speed of: " << _modePathSpeedList[_currentMode] << std::endl;
            PluginManager::instance()->sendMessageByName(PathPlugin,PR_SET_PLAYBACK_SPEED,(char*)&_modePathSpeedList[_currentMode]);
	    PluginManager::instance()->sendMessageByName(PathPlugin,PR_START,NULL);
	    
	    break;
	}
    }
}

void TourCave::message(int type, char * & data, bool)
{
    if(type == PR_GET_TIME)
    {
	_currentTime = *((double*)data);
    }
    else if(type == PR_GET_START_MAT)
    {
	_startMat = osg::Matrix((osg::Matrix::value_type*)data);
    }
    else if(type == PR_GET_START_SCALE)
    {
	_startScale = *((float*)data);
    }
    else if(type == PR_IS_STOPPED)
    {
	bool stopped = *((bool*)data);
	if(_currentMode >= 0 && stopped)
	{
	    for(int j = 0; j < _modeAudioList[_currentMode].size(); j++)
	    {
		if(ComController::instance()->isMaster())
		{
		    // stop audio
#ifdef HAS_AUDIO
		    if(_modeAudioList[_currentMode][j]->sl == PATH_END)
		    {
			if(_modeAudioList[_currentMode][j]->sa == STOP)
			{
			    std::cerr << "Stopping audio file: " << _modeAudioList[_currentMode][j]->name << std::endl;
			    _am->SendUDP((_modeAudioList[_currentMode][j]->name + "/Stop").c_str(),"i", 1);
			}
			else if(_modeAudioList[_currentMode][j]->sa == PAUSE)
			{
			    std::cerr << "Pause audio file: " << _modeAudioList[_currentMode][j]->name << std::endl;
			    _am->SendUDP((_modeAudioList[_currentMode][j]->name + "/Pause").c_str(),"i", 1);
			}
		    }
#endif
		}
	    }
	    _mStatus = DONE;	    
	}
    }
}

void TourCave::checkSockets()
{
    if(!_socketList.size())
    {
	return;
    }

    int maxfd = 0;

    fd_set socketsetR;
    FD_ZERO(&socketsetR);

    for(int i = 0; i < _socketList.size(); i++)
    {
	FD_SET((unsigned int)_socketList[i]->getSocketFD(),&socketsetR);
	if(_socketList[i]->getSocketFD() > maxfd)
	{
	    maxfd = _socketList[i]->getSocketFD();
	}
    }

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;

    select(maxfd+1,&socketsetR,NULL,NULL,&tv);

    for(std::vector<CVRSocket*>::iterator it = _socketList.begin(); it != _socketList.end(); )
    {
	if(FD_ISSET((*it)->getSocketFD(),&socketsetR))
	{
	    if(!processSocketInput(*it))
	    {
		std::cerr << "Removing socket." << std::endl;
                delete *it;
		it = _socketList.erase(it);
	    }
	    else
	    {
		it++;
	    }
	}
	else
	{
	    it++;
	}
    }
}

bool TourCave::processSocketInput(CVRSocket * socket)
{
    char c;
    if(!socket->recv(&c,sizeof(char)))
    {
	return false;
    }

    std::cerr << "char: " << c << std::endl;
    _commandList.push_back(c);

    return true;
}

void TourCave::processCommands(char * commands, int size)
{
    for(int i = 0; i < size; i++)
    {
	int mode = (int)(commands[i] - '0');

	if(mode >= 0 && mode < _modeButtonList.size())
	{
	    menuCallback(_modeButtonList[mode]);
	}
	else
	{
	    std::cerr << "Mode: " << mode << std::endl;
	    bool b;
	    if(mode == 4 && ComController::instance()->isMaster())
	    {
#ifdef HAS_AUDIO
		_am->SetVolume(1.0);
#endif
	    }
	    else if(mode == 5 && ComController::instance()->isMaster())
	    {
#ifdef HAS_AUDIO
		_am->SetVolume(0.0);
#endif
	    }
	    else if(mode == 6)
	    {
		b = true;
		PluginManager::instance()->sendMessageByName("MenuBasics",MB_STEREO,(char*)&b);
	    }
	    else if(mode == 7)
	    {
		b = false;
		PluginManager::instance()->sendMessageByName("MenuBasics",MB_STEREO,(char*)&b);
	    }
	    else if(mode == 8)
	    {
		b = true;
		PluginManager::instance()->sendMessageByName("MenuBasics",MB_HEAD_TRACKING,(char*)&b);
	    }
	    else if(mode == 9)
	    {
		b = false;
		PluginManager::instance()->sendMessageByName("MenuBasics",MB_HEAD_TRACKING,(char*)&b);
	    }
	}
    }
}
