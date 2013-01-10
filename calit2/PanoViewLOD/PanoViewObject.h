#ifndef PANOVIEW_OBJECT_H
#define PANOVIEW_OBJECT_H

#include "PanoDrawableLOD.h"

#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>

#include <osg/Uniform>

#include <string>

struct ZoomTransitionInfo
{
    float rotationFromImage;
    float rotationToImage;
    float zoomValue;
    osg::Vec3 zoomDir;
};

struct MorphTransitionInfo
{
};

class PanoViewObject : public cvr::SceneObject
{
    public:
        PanoViewObject(std::string name, std::string leftEyeFile, std::string rightEyeFile, float radius, int mesh, int depth, int size, float height, std::string vertFile = "sph-zoomer.vert", std::string fragFile = "sph-render.frag");
        PanoViewObject(std::string name, std::vector<std::string> & leftEyeFiles, std::vector<std::string> & rightEyeFiles, float radius, int mesh, int depth, int size, float height, std::string vertFile = "sph-zoomer.vert", std::string fragFile = "sph-render.frag");
        virtual ~PanoViewObject();

        void init(std::vector<std::string> & leftEyeFiles, std::vector<std::string> & rightEyeFiles, float radius, int mesh, int depth, int size, float height, std::string vertFile, std::string fragFile);

        void setTransition(PanTransitionType transitionType, std::string transitionFilesDir, std::vector<std::string> & leftTransitionFiles, std::vector<std::string> & rightTransitionFiles, std::string configTag);

        void next();
        void previous();

        void setAlpha(float alpha);
        float getAlpha();

        void setRotate(float rotate);
        float getRotate();

        void setRemoveOnClick(bool b)
        {
            _removeOnClick = b;
        }

        virtual void menuCallback(cvr::MenuItem * item);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual bool eventCallback(cvr::InteractionEvent * ie);

        void preFrameUpdate();

    protected:
        void updateZoom(osg::Matrix & mat);
        void startTransition();

        bool _printValues;
        bool _removeOnClick;

        PanTransitionType _transitionType;
        std::vector<std::string> _leftTransitionFiles;
        std::vector<std::string> _rightTransitionFiles;
        std::string _transitionFilesDir;

        std::vector<ZoomTransitionInfo> _zoomTransitionInfo;

        osg::Geode * _leftGeode;
        osg::Geode * _rightGeode;
        PanoDrawableLOD * _rightDrawable;
        PanoDrawableLOD * _leftDrawable;

        std::string _imageSearchPath;
        float _floorOffset;

        float _currentZoom;

        osg::Matrix _heightMat;
        osg::Matrix _spinMat;
        osg::Matrix _tiltMat;
        osg::Matrix _coordChangeMat;

        osg::Matrix _lastZoomMat;

        osg::Vec3 _offset;

        osg::Vec3 _tbDir;
        bool _tbDirValid;
        int _tbHand;

        cvr::MenuButton * _nextButton;
        cvr::MenuButton * _previousButton;
        cvr::MenuRangeValue * _radiusRV;
        cvr::MenuRangeValue * _heightRV;
        cvr::MenuRangeValue * _alphaRV;
        cvr::MenuCheckbox * _spinCB;
        cvr::MenuCheckbox * _zoomCB;
        cvr::MenuButton * _zoomResetButton;
        cvr::MenuCheckbox * _demoMode;
        cvr::MenuCheckbox * _trackball;

        osg::ref_ptr<osg::Uniform> _alphaUni;

        double _demoTime;
        double _demoChangeTime;

        float _spinScale;
        float _zoomScale;
        int _zoomValuator;
        int _spinValuator;
        bool _sharedValuator;

        bool _fadeActive;
        int _fadeFrames;

        PanoDrawableInfo * _pdi;

        int _transitionSkipFrames;
        float _transitionTime;
        bool _transitionStarted;
        bool _rotateDone;
        bool _zoomDone;
        bool _fadeDone;

        float _rotateStartDelay;
        float _rotateInterval;
        float _rotateStart;
        float _rotateEnd;
        float _finalRotate;
        float _rotateRampUp;
        float _rotateRampDown;
        float _rotateAmp;

        float _zoomStartDelay;
        float _zoomInterval;
        float _zoomEnd;
        float _zoomRampUp;
        float _zoomRampDown;
        float _zoomAmp;
        osg::Vec3 _zoomTransitionDir;
        
        float _fadeStartDelay;
        float _fadeInterval; 

        bool _upArrow;
        bool _downArrow;
        bool _leftArrow;
        bool _rightArrow;
};

#endif
