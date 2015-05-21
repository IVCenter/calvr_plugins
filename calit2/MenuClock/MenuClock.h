#ifndef MENU_CLOCK_H
#define MENU_CLOCK_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuItemGroup.h>
#include <cvrMenu/MenuText.h>

#include <string>

class MenuClock : public cvr::CVRPlugin
{
    public:
        MenuClock();
        virtual ~MenuClock();

        bool init();
        void preFrame();

    protected:
        void updateClock();

        cvr::MenuItemGroup * _itemGroup;
        cvr::MenuText * _clockText;

        std::string _clockStr;
};

#endif
