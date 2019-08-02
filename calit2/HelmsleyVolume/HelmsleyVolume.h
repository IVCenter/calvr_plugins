#ifndef HELMSLEY_VOLUME_H
#define HELMSLEY_VOLUME_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRadial.h>
#include "VolumeDrawable.h"
#include "VolumeGroup.h"

#include <string>

class HelmsleyVolume : public cvr::MenuCallback, public cvr::CVRPlugin
{
    public:
        HelmsleyVolume();
        virtual ~HelmsleyVolume();

        bool init();
        void preFrame();
		void postFrame();
		void menuCallback(cvr::MenuItem* menuItem);
		void createList(cvr::SubMenu* , std::string configbase);

    protected:
		cvr::SubMenu * _vMenu;
		cvr::MenuButton * _vButton;
		std::map<cvr::MenuItem*, std::string> _buttonMap;
		cvr::MenuRadial * _radial;
};

#endif
