#ifndef CVR_COLLABORATIVE_PLUGIN_H
#define CVR_COLLABORATIVE_PLUGIN_H

#include <kernel/CVRPlugin.h>
#include <menu/MenuButton.h>
#include <menu/SubMenu.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuText.h>

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
