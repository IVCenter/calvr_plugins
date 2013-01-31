#ifndef CVRPLUGIN_PDOBJECT_H
#define CVRPLUGIN_PDOBJECT_H

#include <cvrKernel/SceneObject.h>

class PDObject : public cvr::SceneObject
{
    public:
        PDObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds = false);
        virtual ~PDObject();
};

#endif
