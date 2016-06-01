#ifndef WOUNDVAC_H
#define WOUNDVAC_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrUtil/CVRSocket.h>

#include "SimulationObject.h"

#include <string>

class WoundVAC : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        WoundVAC();
        virtual ~WoundVAC();

        bool init();
        void preFrame();

        void menuCallback(cvr::MenuItem * item);

    protected:
        cvr::CVRSocket * _socket;

        cvr::SubMenu * _menu;
        cvr::MenuCheckbox * _connectCB;

        std::string _host;
        int _port;

        SimulationObject * _simObject;
};

#endif
