#ifndef SIMULATION_OBJECT_H
#define SIMULATION_OBJECT_H

#include <cvrKernel/SceneObject.h>
#include <cvrUtil/CVRSocket.h>

#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/writer.h>

#include <vector>

#include "ShadowObject.h"

class SimulationObject : public cvr::SceneObject, public OpenThreads::Thread
{
    public:
        struct Message
        {
            char type;
            char * data;
            int dataSize;
        };

        SimulationObject(cvr::CVRSocket * socket, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~SimulationObject();

        void preFrame();

        void run();
        void quit();

    protected:
        void createOrigin();
        void createDevice();
        void createSheet();
        void createSponge();

        void checkSocket();
        void processMessages();
        void sendMessages();

        void sendUpdate();
        void updateState();
        void updateLines();

        Json::Value _jsonRoot;
        Json::Value _jsonLocalRoot;
        Json::StreamWriterBuilder _jsonWriterFactory;

        cvr::CVRSocket * _socket;

        ShadowObject * _deviceObject;
        osg::ref_ptr<osg::MatrixTransform> _deviceXForm;
        osg::ref_ptr<osg::Vec3Array> _deviceLine;

        ShadowObject * _sheetObject;
        osg::ref_ptr<osg::MatrixTransform> _sheetXForm;
        osg::ref_ptr<osg::Vec3Array> _sheetLine;

        ShadowObject * _spongeObject;
        osg::ref_ptr<osg::MatrixTransform> _spongeXForm;
        osg::ref_ptr<osg::Vec3Array> _spongeLine;

        osg::ref_ptr<osg::Geode> _lineGeode;

        OpenThreads::Mutex _updateLock;
        bool _updateReady;
        bool _sendReady;
        osg::Vec3 _devPos;
        osg::Quat _devRot;
        osg::Vec3 _shPos;
        osg::Quat _shRot;
        osg::Vec3 _spPos;
        osg::Quat _spRot;

        osg::Vec3 _handPos;
        osg::Quat _handRot;

        int _handId;

        OpenThreads::Mutex _quitLock;
        bool _quit;

        OpenThreads::Mutex _messageLock;
        std::vector<struct Message *> _messageList;
};

#endif
