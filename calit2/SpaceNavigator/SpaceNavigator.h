#ifndef SPACE_NAVIGATOR_H
#define SPACE_NAVIGATOR_H

#include <kernel/CVRPlugin.h>

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <X11/Xlib.h>
#include <spnav.h>

#include <iostream>

class SpaceNavigator : public cvr::CVRPlugin
{
    public:
        SpaceNavigator();
        ~SpaceNavigator();

        bool init();

        void preFrame();

    protected:
        float transMult, rotMult;
        float transcale, rotscale;
};

#endif
