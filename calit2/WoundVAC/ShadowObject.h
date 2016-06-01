#ifndef SHADOW_OBJECT_H
#define SHADOW_OBJECT_H

#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuButton.h>

#include <OpenThreads/Mutex>

class ShadowObject : public cvr::SceneObject
{
    public:
        ShadowObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~ShadowObject();

        bool isActive()
        {
            _activeLock.lock();
            bool a = _active;
            _activeLock.unlock();
            return a;
        }

        void preFrame();
        void menuCallback(cvr::MenuItem * item);

    protected:
        bool _active;
        OpenThreads::Mutex _activeLock;

        cvr::MenuButton * _resetButton;
};

#endif
