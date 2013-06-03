#include "SoundTest.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/PluginHelper.h>
#include <PluginMessageType.h>
#include <unistd.h>


#include <osg/Matrix>

using namespace osg;
using namespace std;
using namespace cvr;
<<<<<<< HEAD
=======
using namespace sc;
>>>>>>> upstream/master

CVRPLUGIN(SoundTest)

SoundTest::SoundTest()
{
}

bool SoundTest::init()
{
    std::cerr << "SoundTest init\n";

    MLMenu = new SubMenu("SoundTest", "SoundTest");
    MLMenu->setCallback(this);

    loadMenu = new SubMenu("Load","Load");
    loadMenu->setCallback(this);
    MLMenu->addItem(loadMenu);

    MenuSystem::instance()->addMenuItem(MLMenu);
    


    std::string name = "Octo";
    std::string synthDir = "/Users/demo/workspace/git/colliderplusplus/synthdefs/mac";
<<<<<<< HEAD
    _clientServer = new ColliderPlusPlus::Client_Server(name, "132.239.235.169", 
	"57110", synthDir);
    _clientServer->dumpOSC(1);

    std::string filepath = "/Users/demo/Desktop/plucky.wav";
    _sound = new ColliderPlusPlus::Sound(_clientServer, filepath, 0);
    usleep(1000000);
    _sound->loop(1);
=======
    _AudioServer = new SCServer(name, "132.239.235.169", 
	"57110", synthDir);
    _AudioServer->dumpOSC(1);

    std::string filepath = "/Users/demo/Desktop/plucky.wav";
    _sound = new Sound(_AudioServer, filepath, 0);
    usleep(1000000);
    _sound->setLoop(1);
>>>>>>> upstream/master
    _sound->play();

    float x, y, z;
    x = 2000;
    y = 2000;
    z = 0;

    //sound->setPosition(x, y, z);

    std::cerr << "SoundTest init done.\n";
    return true;
}


SoundTest::~SoundTest()
{
//    delete _clientServer;
    delete _sound;
}

void SoundTest::menuCallback(MenuItem* menuItem)
{

}

void SoundTest::preFrame()
{

}

bool SoundTest::processEvent(cvr::InteractionEvent * event)
{

}

