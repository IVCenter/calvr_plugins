#ifndef VROOM_VIEW_PLUGIN_H
#define VROOM_VIEW_PLUGIN_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrUtil/MultiListenSocket.h>

#include <vector>

#include "VVClient.h"

class VroomView : public cvr::CVRPlugin
{
public:
    VroomView();
    virtual ~VroomView();

    bool init();
    void preFrame();
    
protected:
    std::vector<VVClient*> _clientList;
    cvr::MultiListenSocket * _mls;
    
};

#endif
