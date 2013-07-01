#ifndef CVRPLUGIN_PDOBJECT_H
#define CVRPLUGIN_PDOBJECT_H

#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuButton.h>

class PDObject : public cvr::SceneObject
{
    public:
        PDObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds = false);
        virtual ~PDObject();

        virtual void menuCallback(cvr::MenuItem * item);

        void resetPosition();

    protected:
        cvr::MenuButton * _resetPositionButton;
};

#endif
