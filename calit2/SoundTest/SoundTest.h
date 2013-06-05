#ifndef _SOUNDTEST_
#define _SOUNDTEST_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>

#include <cvrConfig/ConfigManager.h>

#include <osg/MatrixTransform>

#include "libcollider/Sound.hpp"
#include "libcollider/SCServer.hpp"

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
        cvr::SubMenu * MLMenu;
        cvr::MenuButton * _playButton, * _pauseButton, * _resetButton, * _stopButton;
	cvr::MenuCheckbox * _loopCheckbox;
	cvr::MenuRangeValue * _volumeRange, * _startPos;
	sc::Sound * _sound;
	sc::SCServer * _AudioServer;
};

#endif
