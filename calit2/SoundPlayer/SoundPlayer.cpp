#include "SoundPlayer.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/PluginHelper.h>
#include <PluginMessageType.h>
#include <unistd.h>


#include <osg/Matrix>

using namespace osg;
using namespace std;
using namespace cvr;
using namespace sc;

CVRPLUGIN(SoundPlayer)

SoundPlayer::SoundPlayer()
{
}

bool SoundPlayer::init()
{ 
    std::cerr << "SoundPlayer init\n";

    MLMenu = new SubMenu("SoundPlayer", "SoundPlayer");
    MLMenu->setCallback(this);

    loadMenu = new SubMenu("Load","Load");
    loadMenu->setCallback(this);
    MLMenu->addItem(loadMenu);

    MenuSystem::instance()->addMenuItem(MLMenu);

    _volumeSlider = new MenuRangeValue("Volume", 0, 1, 0.01);
    _volumeSlider->setCallback(this);
    MLMenu->addItem(_volumeSlider);

    std::string name = ConfigManager::getEntry("Plugin.SoundPlayer.Server.Name");
    std::string synthDir = ConfigManager::getEntry("Plugin.SoundPlayer.Server.SynthDir");
    std::string port = ConfigManager::getEntry("Plugin.SoundPlayer.Server.Port");
    std::string address = ConfigManager::getEntry("Plugin.SoundPlayer.Server.Address");

    if (cvr::ComController::instance()->isMaster())
    {
       // Address and port hacked in - to be fixed 
        _AudioServer = new SCServer(name, "137.110.116.10", "57110", synthDir);
        _AudioServer->dumpOSC(1);
    }
    
    std::string dataDir = ConfigManager::getEntry("Plugin.SoundPlayer.Files");

    std::vector<std::string> filenames;
    ConfigManager::getChildren("Plugin.SoundPlayer.Files", filenames);

    for (int i = 0; i < filenames.size(); ++i)
    {
	std::string name = ConfigManager::getEntry("Plugin.SoundPlayer.Files." + filenames[i]);
        std::string path = dataDir + "/" + name;
        if (cvr::ComController::instance()->isMaster())
        {
	    int outarray [] = {0, 7};
	    Sound * sound = new Sound(_AudioServer, path, outarray, 0);

	    if(sound->isValid())
	    {
		sound->setGain(_volumeSlider->getValue());
		_sounds.push_back(sound);
	    }

	    }
	    else
	    {
	        _sounds.push_back(NULL);
		std::cout << "Unable to load sound " << path << std::endl;
	    }
        

	MenuButton * button = new MenuButton(filenames[i]);
	button->setCallback(this);
	loadMenu->addItem(button);
	_soundButtons.push_back(button);
    }

    std::cerr << "SoundPlayer init done.\n";
    return true;
}

SoundPlayer::~SoundPlayer()
{
    if (cvr::ComController::instance()->isMaster())
    {
	for (int i = 0; i < _sounds.size(); ++i)
	{
	    if (_sounds[i])
		delete _sounds[i];
	}
	delete _AudioServer;
    }
}

void SoundPlayer::menuCallback(MenuItem* menuItem)
{
    for (int i = 0; i < _soundButtons.size(); ++i)
    {
	if (menuItem == _soundButtons[i])
	{
	    if (_AudioServer)
	    {
		if (_sounds[i])
		    _sounds[i]->play();
	    }
	}
    }

    if (menuItem == _volumeSlider)
    {
	if (_AudioServer)
	{
	    for (int i = 0; i < _sounds.size(); ++i)
	    {
		if (_sounds[i])
		{
		    _sounds[i]->setGain(_volumeSlider->getValue());
		}
	    }
	}
    }
}

void SoundPlayer::preFrame()
{

}

bool SoundPlayer::processEvent(cvr::InteractionEvent * event)
{
	return false;
}

