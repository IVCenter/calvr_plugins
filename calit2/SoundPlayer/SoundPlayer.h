#ifndef _SOUNDPLAYER_
#define _SOUNDPLAYER_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrKernel/ComController.h>

#include <osg/MatrixTransform>

#include "Buffer.hpp"
#include "SCServer.hpp"

#include <string>
#include <vector>
#include <map>

class SoundPlayer: public cvr::MenuCallback, public cvr::CVRPlugin
{
    public:        
        SoundPlayer();
        virtual ~SoundPlayer();

	bool init();
        void menuCallback(cvr::MenuItem * item);
        void preFrame();
	bool processEvent(cvr::InteractionEvent * event);

    protected:
        cvr::SubMenu * MLMenu, * loadMenu;
        cvr::MenuButton * removeButton, *oneBtn, *twoBtn, *threeBtn, *vroomBtn;
        cvr::MenuRangeValue *_volumeSlider;
        sc::Buffer *oneSound, *twoSound, *threeSound, *vroomSound;
	sc::SCServer * _AudioServer;
        std::map<std::string, float> oneSoundArgs,twoSoundArgs,threeSoundArgs,vroomSoundArgs;
};

#endif
