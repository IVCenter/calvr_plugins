#include "SoundTest.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/PluginHelper.h>
#include <PluginMessageType.h>
#include <unistd.h>
#include <stdio.h>

#include <osg/Matrix>

using namespace osg;
using namespace std;
using namespace cvr;
using namespace sc;

CVRPLUGIN(SoundTest)

SoundTest::SoundTest()
{


}

bool SoundTest::init()
{
    std::cerr << "SoundTest init\n";

    std::string name = ConfigManager::getEntry("ColliderConfig.ServerName");
    std::string synthDefDir = ConfigManager::getEntry("ColliderConfig.SynthDefDir");
    std::string port = ConfigManager::getEntry("ColliderConfig.ServerPort");
    std::string host = ConfigManager::getEntry("ColliderConfig.ServerIp");

    _AudioServer = new SCServer(name, "127.0.0.1", "57110", synthDefDir);
    _AudioServer->dumpOSC(1); // Commands the remote server to print incoming messages

    std::string filename = "bermuda48.wav";
    std::string fullfilepath = ConfigManager::getEntry("ColliderConfig.ResourceDir")+filename;

    int outarray [] = {0, 1};

    _sound = new Sound(_AudioServer, fullfilepath, outarray, 0);

    if(!(_sound->isValid()))
       std::cerr << "\nError: Sound -> " << fullfilepath
	 << " is not valid. Unable to create Sound." << std::endl;
        
      	MLMenu = new SubMenu("SoundTest", "SoundTest");
    	MLMenu->setCallback(this);

   	 _playButton = new MenuButton("Play");
    	_playButton->setCallback(this);
    	MLMenu->addItem(_playButton);

   	 _pauseButton = new MenuButton("Pause");
   	 _pauseButton->setCallback(this);
    	MLMenu->addItem(_pauseButton);

    	_stopButton = new MenuButton("Stop");
    	_stopButton->setCallback(this);
   	 MLMenu->addItem(_stopButton);

    	_resetButton = new MenuButton("Reset to Beginning");
    	_resetButton->setCallback(this);
    	MLMenu->addItem(_resetButton);

    	_loopCheckbox = new MenuCheckbox("Loop", true);
    	_loopCheckbox->setCallback(this);
    	MLMenu->addItem(_loopCheckbox);

    	_volumeRange = new MenuRangeValue("Volume", 0.0f, 1.0f, 0.5f);
    	_volumeRange->setCallback(this);
    	MLMenu->addItem(_volumeRange);

    	_startPos = new MenuRangeValue("Start Position", 0.0f, 1.0f, 0.0f);
    	_startPos->setCallback(this);
    	MLMenu->addItem(_startPos);

    	PluginHelper::addRootMenuItem(MLMenu);

    	_sound->setGain(_volumeRange->getValue()); 
    	_sound->setStartPosition(_startPos->getValue());
    	_sound->setLoop(true);
	std::cerr << "\nSoundTest init done.\n";
    
     return true;
}


SoundTest::~SoundTest()
{
    if(_sound)
    	delete _sound;

    if(_AudioServer)
    {
        _AudioServer->dumpOSC(0);
    	delete _AudioServer;
    }
}

void SoundTest::menuCallback(MenuItem* item)
{
    if(item == _playButton)
    {
         _sound->play();
    }

    else if(item == _pauseButton)
    {
	 _sound->pause();
    }

    else if(item == _loopCheckbox)
    {
	 if(_loopCheckbox->getValue())
              	_sound->setLoop(true);
	 else
	      	_sound->setLoop(false);
    }

    else if(item == _stopButton)
    {
	if(_sound->isPlaying())
	{
	 	_sound->jumpToStartPos();
	 	_sound->pause();
	}
    }

    else if(item == _resetButton)
    {
	if(_sound->isPlaying())
        {
		_startPos->setValue(0.0f);
		_sound->setStartPosition(_startPos->getValue());
	 	_sound->jumpToStartPos();
		_sound->pause();
	}
    }

    else if(item == _volumeRange)
    {
         _sound->setGain(_volumeRange->getValue());
    }

    else if(item == _startPos)
    { 	 
         _sound->setStartPosition(_startPos->getValue()); 
    }
}

void SoundTest::preFrame()
{

}

bool SoundTest::processEvent(cvr::InteractionEvent * event)
{
  return false;
}

