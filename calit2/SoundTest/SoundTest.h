#ifndef _SOUNDTEST_
#define _SOUNDTEST_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>

#include <osg/MatrixTransform>

<<<<<<< HEAD
#include "ColliderPlusPlus/Sound.hpp"
#include "ColliderPlusPlus/Client_Server.hpp"
=======
#include "libcollider/Sound.hpp"
#include "libcollider/SCServer.hpp"
>>>>>>> upstream/master

#include <string>
#include <vector>

class SoundTest : public cvr::MenuCallback, public cvr::CVRPlugin
{
    public:        
        SoundTest();
        virtual ~SoundTest();

	bool init();
        void menuCallback(cvr::MenuItem * item);
        void preFrame();
	bool processEvent(cvr::InteractionEvent * event);

    protected:
        cvr::SubMenu * MLMenu, * loadMenu;
        cvr::MenuButton * removeButton;
<<<<<<< HEAD
	ColliderPlusPlus::Sound * _sound;
	ColliderPlusPlus::Client_Server * _clientServer;
=======
	sc::Sound * _sound;
	sc::SCServer * _AudioServer;
>>>>>>> upstream/master
};

#endif
