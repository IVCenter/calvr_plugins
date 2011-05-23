#ifndef CVR_COLLABORATIVE_PLUGIN_H
#define CVR_COLLABORATIVE_PLUGIN_H

#include <kernel/CVRPlugin.h>
#include <menu/MenuButton.h>
#include <menu/SubMenu.h>
#include <menu/MenuCheckbox.h>

#include <string>

class Collaborative : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        Collaborative();
        ~Collaborative();

        bool init();

        void menuCallback(cvr::MenuItem * item);

    protected:
        std::string _server;
        int _port;

        cvr::SubMenu * _collabMenu;

        cvr::MenuCheckbox * _enable;
};

#endif
