#include <kernel/CVRPlugin.h>
#include <kernel/Navigation.h>

#include <iostream>

#include <osg/MatrixTransform>
#include <osg/Matrix>

class MultiHandInteraction : public cvr::CVRPlugin
{
    public:
        MultiHandInteraction();
        ~MultiHandInteraction();

        bool init();
        void preFrame();

        int getPriority() { return 1; }

        bool processEvent(cvr::InteractionEvent * event);

    protected:
        void processNav();
        void newProcessNav();

        bool _interactionStarted;
        cvr::NavMode _navMode;
        int _activeHand;
        int _refHand;
        osg::Matrix _refHandMat;
        osg::Matrix _activeHandMat;

        osg::Matrix _startXForm;

        // scale
        float _startScale;
        float _startScaleLength;
        osg::Vec3 _scalePoint;

        // drive,fly
        osg::Vec3 _startPoint;
        osg::Quat _startRot;

        // new stuff
        osg::Matrix _lastRefHandMat;
        osg::Matrix _lastActiveHandMat;
        osg::Matrix _currentRefHandMat;
        osg::Matrix _currentActiveHandMat;

        bool _refUpdated;
        bool _activeUpdated;
        bool _setLastRefHand;
};
