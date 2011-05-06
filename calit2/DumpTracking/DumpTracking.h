#ifndef DUMP_TRACKING_H
#define DUMP_TRACKING_H

#include <kernel/CVRPlugin.h>
#include <iostream>
#include <fstream>

class DumpTracking : public cvr::CVRPlugin
{
    public:
        DumpTracking();
        ~DumpTracking();

        bool init();
        void preFrame();
    protected:
        std::fstream _file;
        char * _stringOut;
};

#endif
