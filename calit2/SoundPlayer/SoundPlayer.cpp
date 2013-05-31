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
    
    oneBtn = new MenuButton("Sound 1");
    oneBtn->setCallback(this);
    loadMenu->addItem(oneBtn);

    twoBtn = new MenuButton("Sound 2");
    twoBtn->setCallback(this);
    loadMenu->addItem(twoBtn);

    threeBtn = new MenuButton("Sound 3");
    threeBtn->setCallback(this);
    loadMenu->addItem(threeBtn);

    vroomBtn = new MenuButton("Vroom");
    vroomBtn->setCallback(this);
    loadMenu->addItem(vroomBtn);

    _volumeSlider = new MenuRangeValue("Volume", 0, 1, 0.5);
    _volumeSlider->setCallback(this);
    MLMenu->addItem(_volumeSlider);

    std::string name = "Octo";
    std::string synthDir = "/Users/demo/Desktop/workspace/svn/SonicaveHMC/libcollider";


    if (cvr::ComController::instance()->isMaster())
    {
	_AudioServer = new SCServer(name, "137.110.116.10", 
	    "57110", synthDir);
	_AudioServer->dumpOSC(1);

	std::string filepath = "/Users/demo/Desktop/plucky.wav";
	//_sound = new Sound(_AudioServer, filepath, 0);
	//usleep(1000000);
	//_sound->setLoop(1);
	//_sound->play();
	 
	filepath = "/Users/demo/Desktop/workspace/svn/SonicaveHMC/soundfiles/CAVEsnd1.wav";
	oneSound = new Buffer(_AudioServer, _AudioServer->nextBufferNum());
	oneSound->allocRead(filepath);

	filepath = "/Users/demo/Desktop/workspace/svn/SonicaveHMC/soundfiles/CAVEsnd2.wav";
	twoSound = new Buffer(_AudioServer, _AudioServer->nextBufferNum());
	twoSound->allocRead(filepath);

	filepath = "/Users/demo/Desktop/workspace/svn/SonicaveHMC/soundfiles/CAVEsnd3.wav";
	threeSound = new Buffer(_AudioServer, _AudioServer->nextBufferNum());
	threeSound->allocRead(filepath);

	filepath = "/Users/demo/Desktop/workspace/svn/SonicaveHMC/soundfiles/InstLIST_VROOM.wav";
	vroomSound = new Buffer(_AudioServer, _AudioServer->nextBufferNum());
	vroomSound->allocRead(filepath);

	oneSoundArgs["bufnum"] = oneSound->getBufNum();
	twoSoundArgs["bufnum"] = twoSound->getBufNum();
	threeSoundArgs["bufnum"] = threeSound->getBufNum();
	vroomSoundArgs["bufnum"] = vroomSound->getBufNum();
    }

    float x, y, z;
    x = 2000;
    y = 2000;
    z = 0;

    //sound->setPosition(x, y, z);

    std::cerr << "SoundPlayer init done.\n";
    return true;
}


SoundPlayer::~SoundPlayer()
{
//    delete _clientServer;
    //delete _sound;
    if (cvr::ComController::instance()->isMaster())
    {
	delete oneSound;
	delete twoSound;
	delete threeSound;
	delete vroomSound;
	delete _AudioServer;
    }
}

void SoundPlayer::menuCallback(MenuItem* menuItem)
{
    if (menuItem == oneBtn)
    {
	if (cvr::ComController::instance()->isMaster())
	{
	    _AudioServer->createSynth("SoundFile_Event_Stereo_81", 1000,oneSoundArgs);
	}
    }
    else if (menuItem == twoBtn)
    {
    if (cvr::ComController::instance()->isMaster())
	{
	    _AudioServer->createSynth("SoundFile_Event_Stereo_81", 1001,twoSoundArgs);
	}
    }
    else if (menuItem == threeBtn)
    {
	if (cvr::ComController::instance()->isMaster())
	{
	    _AudioServer->createSynth("SoundFile_Event_Stereo_81", 1002,threeSoundArgs);
	}
    }
    else if (menuItem == vroomBtn)
    {
	if (cvr::ComController::instance()->isMaster())
	{
	    _AudioServer->createSynth("SoundFile_Event_Stereo_81", 1003,vroomSoundArgs);
	}
    }
    else if (menuItem == _volumeSlider)
    {
	if (cvr::ComController::instance()->isMaster())
	{
	oneSoundArgs["amp"] = _volumeSlider->getValue();
	_AudioServer->setNodeControls(1000, oneSoundArgs);

	twoSoundArgs["amp"] = _volumeSlider->getValue();
	_AudioServer->setNodeControls(1001, oneSoundArgs);

	threeSoundArgs["amp"] = _volumeSlider->getValue();
	_AudioServer->setNodeControls(1002, oneSoundArgs);

	vroomSoundArgs["amp"] = _volumeSlider->getValue();
	_AudioServer->setNodeControls(1003, oneSoundArgs);
	}
    }
}

void SoundPlayer::preFrame()
{

}

bool SoundPlayer::processEvent(cvr::InteractionEvent * event)
{

}

