#ifndef CVR_COLLABORATIVE_PLUGIN_H
#define CVR_COLLABORATIVE_PLUGIN_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuText.h>

#include <string>
#include <map>

class Collaborative : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        Collaborative();
        ~Collaborative();

        bool init();

        void menuCallback(cvr::MenuItem * item);

        void preFrame();

    protected:
        void addMenuItems();
        void removeMenuItems();
        void updateMenuItems();

        std::string _server;
        int _port;

        cvr::SubMenu * _collabMenu;

        cvr::MenuCheckbox * _enable;
        cvr::MenuCheckbox * _lockedCB;
        cvr::MenuCheckbox * _myCB;
        cvr::MenuText * _clientText;

        std::map<int,cvr::MenuCheckbox*> _clientCBMap;
};

#endif
