#include "CustomStatsHandler.h"

#include <iostream>
#include <sstream>
#include <iomanip>

#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/Renderer>

#include <osg/PolygonMode>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osgText/Text>

using namespace osgViewer;

struct AveragedValueTextDrawCallback : public virtual osg::Drawable::DrawCallback
{
    AveragedValueTextDrawCallback(osg::Stats* stats, const std::string& name, int frameDelta, bool averageInInverseSpace, double multiplier):
        _stats(stats),
        _attributeName(name),
        _frameDelta(frameDelta),
        _averageInInverseSpace(averageInInverseSpace),
        _multiplier(multiplier),
        _tickLastUpdated(0)
    {
    }

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo& renderInfo,const osg::Drawable* drawable) const
    {
        osgText::Text* text = (osgText::Text*)drawable;

        osg::Timer_t tick = osg::Timer::instance()->tick();
        double delta = osg::Timer::instance()->delta_m(_tickLastUpdated, tick);

        if (delta>50) // update every 50ms
        {
            _tickLastUpdated = tick;
            double value;
            if (_stats->getAveragedAttribute( _attributeName, value, _averageInInverseSpace))
            {
                sprintf(_tmpText,"%4.2f",value * _multiplier);
                text->setText(_tmpText);
            }
            else
            {
                text->setText("");
            }
        }
        text->drawImplementation(renderInfo);
    }

    osg::ref_ptr<osg::Stats>    _stats;
    std::string                 _attributeName;
    int                         _frameDelta;
    bool                        _averageInInverseSpace;
    double                      _multiplier;
    mutable char                _tmpText[128];
    mutable osg::Timer_t        _tickLastUpdated;
};

struct ViewSceneStatsTextDrawCallback : public virtual osg::Drawable::DrawCallback
{
    ViewSceneStatsTextDrawCallback(osgViewer::View* view, int viewNumber):
        _view(view),
        _tickLastUpdated(0),
        _viewNumber(viewNumber)
    {
    }

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo& renderInfo,const osg::Drawable* drawable) const
    {
        if (!_view) return;

        osgText::Text* text = (osgText::Text*)drawable;

        osg::Timer_t tick = osg::Timer::instance()->tick();
        double delta = osg::Timer::instance()->delta_m(_tickLastUpdated, tick);

        if (delta > 200) // update every 100ms
        {
            _tickLastUpdated = tick;
            osg::Stats* stats = _view->getStats();
            if (stats)
            {
                std::ostringstream viewStr;
                viewStr.clear();
                viewStr.setf(std::ios::left, std::ios::adjustfield);
                viewStr.width(20);
                viewStr.setf(std::ios::fixed);
                viewStr.precision(0);

                viewStr << std::setw(1) << "#" << _viewNumber;

                // View name
                if (!_view->getName().empty())
                    viewStr << ": " << _view->getName();
                viewStr << std::endl;

                int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();
                // if (!(renderer->getGraphicsThreadDoesCull()))
                {
                    --frameNumber;
                }

                #define STATS_ATTRIBUTE_PAIR(str1, str2) \
                    if (stats->getAttribute(frameNumber, str1, value)) \
                        viewStr << std::setw(9) << value; \
                    else \
                        viewStr << std::setw(9) << "."; \
                    if (stats->getAttribute(frameNumber, str2, value)) \
                        viewStr << std::setw(9) << value << std::endl; \
                    else \
                        viewStr << std::setw(9) << "." << std::endl; \

                double value = 0.0;

                // header
                viewStr << std::setw(9) << "Unique" << std::setw(9) << "Instance" << std::endl;

                STATS_ATTRIBUTE_PAIR("Number of unique StateSet","Number of instanced Stateset")
                STATS_ATTRIBUTE_PAIR("Number of unique Group","Number of instanced Group")
                STATS_ATTRIBUTE_PAIR("Number of unique Transform","Number of instanced Transform")
                STATS_ATTRIBUTE_PAIR("Number of unique LOD","Number of instanced LOD")
                STATS_ATTRIBUTE_PAIR("Number of unique Switch","Number of instanced Switch")
                STATS_ATTRIBUTE_PAIR("Number of unique Geode","Number of instanced Geode")
                STATS_ATTRIBUTE_PAIR("Number of unique Drawable","Number of instanced Drawable")
                STATS_ATTRIBUTE_PAIR("Number of unique Geometry","Number of instanced Geometry")
                STATS_ATTRIBUTE_PAIR("Number of unique Vertices","Number of instanced Vertices")
                STATS_ATTRIBUTE_PAIR("Number of unique Primitives","Number of instanced Primitives")


                text->setText(viewStr.str());
            }
            else
            {
                //OSG_NOTIFY(osg::WARN)<<std::endl<<"No valid view to collect scene stats from"<<std::endl;

                text->setText("");
            }
        }
        text->drawImplementation(renderInfo);
    }

    osg::observer_ptr<osgViewer::View>  _view;
    mutable osg::Timer_t                _tickLastUpdated;
    int                                 _viewNumber;
};

struct CameraSceneStatsTextDrawCallback : public virtual osg::Drawable::DrawCallback
{
    CameraSceneStatsTextDrawCallback(osg::Camera* camera, int cameraNumber):
        _camera(camera),
        _tickLastUpdated(0),
        _cameraNumber(cameraNumber)
    {
    }

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo& renderInfo,const osg::Drawable* drawable) const
    {
        if (!_camera) return;

        osgText::Text* text = (osgText::Text*)drawable;

        osg::Timer_t tick = osg::Timer::instance()->tick();
        double delta = osg::Timer::instance()->delta_m(_tickLastUpdated, tick);

        if (delta > 100) // update every 100ms
        {
            _tickLastUpdated = tick;
            std::ostringstream viewStr;
            viewStr.clear();

            osg::Stats* stats = _camera->getStats();
            osgViewer::Renderer* renderer = dynamic_cast<osgViewer::Renderer*>(_camera->getRenderer());

            if (stats && renderer)
            {
                viewStr.setf(std::ios::left, std::ios::adjustfield);
                viewStr.width(14);
                // Used fixed formatting, as scientific will switch to "...e+.." notation for
                // large numbers of vertices/drawables/etc.
                viewStr.setf(std::ios::fixed);
                viewStr.precision(0);

                viewStr << std::setw(1) << "#" << _cameraNumber << std::endl;

                // Camera name
                if (!_camera->getName().empty())
                    viewStr << _camera->getName();
                viewStr << std::endl;

                int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();
                if (!(renderer->getGraphicsThreadDoesCull()))
                {
                    --frameNumber;
                }

                #define STATS_ATTRIBUTE(str) \
                    if (stats->getAttribute(frameNumber, str, value)) \
                        viewStr << std::setw(8) << value << std::endl; \
                    else \
                        viewStr << std::setw(8) << "." << std::endl; \

                double value = 0.0;

                STATS_ATTRIBUTE("Visible number of lights")
                STATS_ATTRIBUTE("Visible number of render bins")
                STATS_ATTRIBUTE("Visible depth")
                STATS_ATTRIBUTE("Visible number of materials")
                STATS_ATTRIBUTE("Visible number of impostors")
                STATS_ATTRIBUTE("Visible number of drawables")
                STATS_ATTRIBUTE("Visible vertex count")

                STATS_ATTRIBUTE("Visible number of GL_POINTS")
                STATS_ATTRIBUTE("Visible number of GL_LINES")
                STATS_ATTRIBUTE("Visible number of GL_LINE_STRIP")
                STATS_ATTRIBUTE("Visible number of GL_LINE_LOOP")
                STATS_ATTRIBUTE("Visible number of GL_TRIANGLES")
                STATS_ATTRIBUTE("Visible number of GL_TRIANGLE_STRIP")
                STATS_ATTRIBUTE("Visible number of GL_TRIANGLE_FAN")
                STATS_ATTRIBUTE("Visible number of GL_QUADS")
                STATS_ATTRIBUTE("Visible number of GL_QUAD_STRIP")
                STATS_ATTRIBUTE("Visible number of GL_POLYGON")

                text->setText(viewStr.str());
            }
        }
        text->drawImplementation(renderInfo);
    }

    osg::observer_ptr<osg::Camera>  _camera;
    mutable osg::Timer_t            _tickLastUpdated;
    int                             _cameraNumber;
};



struct FrameMarkerDrawCallback : public virtual osg::Drawable::DrawCallback
{
    FrameMarkerDrawCallback(StatsHandler* statsHandler, float xPos, osg::Stats* viewerStats, int frameDelta, int numFrames):
        _statsHandler(statsHandler),
        _xPos(xPos),
        _viewerStats(viewerStats),
        _frameDelta(frameDelta),
        _numFrames(numFrames) {}

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo& renderInfo,const osg::Drawable* drawable) const
    {
        osg::Geometry* geom = (osg::Geometry*)drawable;
        osg::Vec3Array* vertices = (osg::Vec3Array*)geom->getVertexArray();

        int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();

        int startFrame = frameNumber + _frameDelta - _numFrames + 1;
        int endFrame = frameNumber + _frameDelta;
        double referenceTime;
        if (!_viewerStats->getAttribute( startFrame, "Reference time", referenceTime))
        {
            return;
        }

        unsigned int vi = 0;
        double currentReferenceTime;
        for(int i = startFrame; i <= endFrame; ++i)
        {
            if (_viewerStats->getAttribute( i, "Reference time", currentReferenceTime))
            {
                (*vertices)[vi++].x() = _xPos + (currentReferenceTime - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (currentReferenceTime - referenceTime) * _statsHandler->getBlockMultiplier();
            }
        }

        drawable->drawImplementation(renderInfo);
    }

    StatsHandler*               _statsHandler;
    float                       _xPos;
    osg::ref_ptr<osg::Stats>    _viewerStats;
    std::string                 _endName;
    int                         _frameDelta;
    int                         _numFrames;
};

struct PagerCallback : public virtual osg::NodeCallback
{

    PagerCallback(    osgDB::DatabasePager* dp,
                    osgText::Text* minValue,
                    osgText::Text* maxValue,
                    osgText::Text* averageValue,
                    osgText::Text* filerequestlist,
                    osgText::Text* compilelist,
                    double multiplier):
        _dp(dp),
        _minValue(minValue),
        _maxValue(maxValue),
        _averageValue(averageValue),
        _filerequestlist(filerequestlist),
        _compilelist(compilelist),
        _multiplier(multiplier)
    {
    }

    virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
    {
        if (_dp.valid())
        {
            double value = _dp->getAverageTimeToMergeTiles();
            if (value>= 0.0 && value <= 1000)
            {
                sprintf(_tmpText,"%4.0f",value * _multiplier);
                _averageValue->setText(_tmpText);
            }
            else
            {
                _averageValue->setText("");
            }

            value = _dp->getMinimumTimeToMergeTile();
            if (value>= 0.0 && value <= 1000)
            {
                sprintf(_tmpText,"%4.0f",value * _multiplier);
                _minValue->setText(_tmpText);
            }
            else
            {
                _minValue->setText("");
            }

            value = _dp->getMaximumTimeToMergeTile();
            if (value>= 0.0 && value <= 1000)
            {
                sprintf(_tmpText,"%4.0f",value * _multiplier);
                _maxValue->setText(_tmpText);
            }
            else
            {
                _maxValue->setText("");
            }

            sprintf(_tmpText,"%4d", _dp->getFileRequestListSize());
            _filerequestlist->setText(_tmpText);

            sprintf(_tmpText,"%4d", _dp->getDataToCompileListSize());
            _compilelist->setText(_tmpText);
        }

        traverse(node,nv);
    }

    osg::observer_ptr<osgDB::DatabasePager> _dp;

    osg::ref_ptr<osgText::Text> _minValue;
    osg::ref_ptr<osgText::Text> _maxValue;
    osg::ref_ptr<osgText::Text> _averageValue;
    osg::ref_ptr<osgText::Text> _filerequestlist;
    osg::ref_ptr<osgText::Text> _compilelist;
    double              _multiplier;
    char                _tmpText[128];
    osg::Timer_t        _tickLastUpdated;
};

struct BlockDrawCallback : public virtual osg::Drawable::DrawCallback
{
    BlockDrawCallback(StatsHandler* statsHandler, float xPos, osg::Stats* viewerStats, osg::Stats* stats, const std::string& beginName, const std::string& endName, int frameDelta, int numFrames):
        _statsHandler(statsHandler),
        _xPos(xPos),
        _viewerStats(viewerStats),
        _stats(stats),
        _beginName(beginName),
        _endName(endName),
        _frameDelta(frameDelta),
        _numFrames(numFrames) {}

    /** do customized draw code.*/
    virtual void drawImplementation(osg::RenderInfo& renderInfo,const osg::Drawable* drawable) const
    {
        osg::Geometry* geom = (osg::Geometry*)drawable;
        osg::Vec3Array* vertices = (osg::Vec3Array*)geom->getVertexArray();

        int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();

        int startFrame = frameNumber + _frameDelta - _numFrames + 1;
        int endFrame = frameNumber + _frameDelta;
        double referenceTime;
        if (!_viewerStats->getAttribute( startFrame, "Reference time", referenceTime))
        {
            return;
        }

        unsigned int vi = 0;
        double beginValue, endValue;
        for(int i = startFrame; i <= endFrame; ++i)
        {
            if (_stats->getAttribute( i, _beginName, beginValue) &&
                _stats->getAttribute( i, _endName, endValue) )
            {
                (*vertices)[vi++].x() = _xPos + (beginValue - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (beginValue - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (endValue - referenceTime) * _statsHandler->getBlockMultiplier();
                (*vertices)[vi++].x() = _xPos + (endValue - referenceTime) * _statsHandler->getBlockMultiplier();
            }
        }

        osg::DrawArrays* drawArrays = static_cast<osg::DrawArrays*>(geom->getPrimitiveSet(0));
        drawArrays->setCount(vi);

        drawable->drawImplementation(renderInfo);
    }

    StatsHandler*               _statsHandler;
    float                       _xPos;
    osg::ref_ptr<osg::Stats>    _viewerStats;
    osg::ref_ptr<osg::Stats>    _stats;
    std::string                 _beginName;
    std::string                 _endName;
    int                         _frameDelta;
    int                         _numFrames;
};

struct StatsGraph : public osg::MatrixTransform
{
    StatsGraph(osg::Vec3 pos, float width, float height)
        : _pos(pos), _width(width), _height(height),
          _statsGraphGeode(new osg::Geode)
    {
        _pos -= osg::Vec3(0, height, 0.1);
        setMatrix(osg::Matrix::translate(_pos));
        addChild(_statsGraphGeode.get());
    }

    void addStatGraph(osg::Stats* viewerStats, osg::Stats* stats, const osg::Vec4& color, float max, const std::string& nameBegin, const std::string& nameEnd = "")
    {
        _statsGraphGeode->addDrawable(new Graph(_width, _height, viewerStats, stats, color, max, nameBegin, nameEnd));
    }

    osg::Vec3           _pos;
    float               _width;
    float               _height;

    osg::ref_ptr<osg::Geode> _statsGraphGeode;

protected:
    struct Graph : public osg::Geometry
    {
        Graph(float width, float height, osg::Stats* viewerStats, osg::Stats* stats,
              const osg::Vec4& color, float max, const std::string& nameBegin, const std::string& nameEnd = "")
        {
            setUseDisplayList(false);

            setVertexArray(new osg::Vec3Array);

            osg::Vec4Array* colors = new osg::Vec4Array;
            colors->push_back(color);
            setColorArray(colors);
            setColorBinding(osg::Geometry::BIND_OVERALL);

            setDrawCallback(new GraphUpdateCallback(width, height, viewerStats, stats, max, nameBegin, nameEnd));
        }
    };

    struct GraphUpdateCallback : public osg::Drawable::DrawCallback
    {
        GraphUpdateCallback(float width, float height, osg::Stats* viewerStats, osg::Stats* stats,
                            float max, const std::string& nameBegin, const std::string& nameEnd = "")
            : _width((unsigned int)width), _height((unsigned int)height), _curX(0),
              _viewerStats(viewerStats), _stats(stats), _max(max), _nameBegin(nameBegin), _nameEnd(nameEnd)
        {
        }

        virtual void drawImplementation(osg::RenderInfo& renderInfo,const osg::Drawable* drawable) const
        {
            osg::Geometry* geometry = const_cast<osg::Geometry*>(drawable->asGeometry());
            if (!geometry) return;
            osg::Vec3Array* vertices = dynamic_cast<osg::Vec3Array*>(geometry->getVertexArray());
            if (!vertices) return;

            int frameNumber = renderInfo.getState()->getFrameStamp()->getFrameNumber();

            // Get stats
            double value;
            if (_nameEnd.empty())
            {
                if (!_stats->getAveragedAttribute( _nameBegin, value, true ))
                {
                    value = 0.0;
                }
            }
            else
            {
                double beginValue, endValue;
                if (_stats->getAttribute( frameNumber, _nameBegin, beginValue) &&
                    _stats->getAttribute( frameNumber, _nameEnd, endValue) )
                {
                    value = endValue - beginValue;
                }
                else
                {
                    value = 0.0;
                }
            }

            // Add new vertex for this frame.
            value = osg::clampTo(value, 0.0, double(_max));
            vertices->push_back(osg::Vec3(float(_curX), float(_height) / _max * value, 0));

            // One vertex per pixel in X.
            if (vertices->size() > _width)
            {
                unsigned int excedent = vertices->size() - _width;
                vertices->erase(vertices->begin(), vertices->begin() + excedent);

                // Make the graph scroll when there is enough data.
                // Note: We check the frame number so that even if we have
                // many graphs, the transform is translated only once per
                // frame.
                static const float increment = -1.0;
                if (GraphUpdateCallback::_frameNumber != frameNumber)
                {
                    // We know the exact layout of this part of the scene
                    // graph, so this is OK...
                    osg::MatrixTransform* transform =
                        geometry->getParent(0)->getParent(0)->asTransform()->asMatrixTransform();
                    if (transform)
                    {
                        transform->setMatrix(transform->getMatrix() * osg::Matrix::translate(osg::Vec3(increment, 0, 0)));
                    }
                }
            }
            else
            {
                // Create primitive set if none exists.
                if (geometry->getNumPrimitiveSets() == 0)
                    geometry->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 0, 0));

                // Update primitive set.
                osg::DrawArrays* drawArrays = dynamic_cast<osg::DrawArrays*>(geometry->getPrimitiveSet(0));
                if (!drawArrays) return;
                drawArrays->setFirst(0);
                drawArrays->setCount(vertices->size());
            }

            _curX++;
            GraphUpdateCallback::_frameNumber = frameNumber;

            geometry->dirtyBound();

            drawable->drawImplementation(renderInfo);
        }

        const unsigned int      _width;
        const unsigned int      _height;
        mutable unsigned int    _curX;
        osg::Stats*             _viewerStats;
        osg::Stats*             _stats;
        const float             _max;
        const std::string       _nameBegin;
        const std::string       _nameEnd;
        static int              _frameNumber;
    };
};

int StatsGraph::GraphUpdateCallback::_frameNumber = 0;

CustomStatsHandler::CustomStatsHandler(int gpus) : StatsHandler()
{
    _numGPUs = gpus;
}

CustomStatsHandler::~CustomStatsHandler()
{
}

bool CustomStatsHandler::handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa)
{

    osgViewer::View* myview = dynamic_cast<osgViewer::View*>(&aa);
    if (!myview) return false;

    osgViewer::ViewerBase* viewer = myview->getViewerBase();
    if (viewer && _threadingModelText.valid() && viewer->getThreadingModel()!=_threadingModel)
    {
        _threadingModel = viewer->getThreadingModel();
        updateThreadingModelText();
    }


    if (ea.getHandled()) return false;

    switch(ea.getEventType())
    {
        case(osgGA::GUIEventAdapter::KEYDOWN):
        {
            if (ea.getKey()==_keyEventTogglesOnScreenStats)
            {
                if (viewer->getViewerStats())
                {
                    if (!_initialized)
                    {
                        setUpHUDCamera(viewer);
                        setUpScene(viewer);
                    }

                    ++_statsType;

                    if (_statsType==LAST) _statsType = NO_STATS;

                    osgViewer::ViewerBase::Cameras cameras;
                    viewer->getCameras(cameras);

                    switch(_statsType)
                    {
                        case(NO_STATS):
                        {
                            viewer->getViewerStats()->collectStats("frame_rate",false);
                            //viewer->getViewerStats()->collectStats("event",false);
                            //viewer->getViewerStats()->collectStats("update",false);

                            for(osgViewer::ViewerBase::Cameras::iterator itr = cameras.begin();
                                itr != cameras.end();
                                ++itr)
                            {
                                osg::Stats* stats = (*itr)->getStats();
                                if (stats)
                                {
				    stats->collectStats("mgpu",false);
                                    //stats->collectStats("rendering",false);
                                    //stats->collectStats("gpu",false);
                                    //stats->collectStats("scene",false);
                                }
                            }

                            viewer->getViewerStats()->collectStats("scene",false);

                            _camera->setNodeMask(0x0);
                            _switch->setAllChildrenOff();
                            break;
                        }
                        case(FRAME_RATE):
                        {
                            viewer->getViewerStats()->collectStats("frame_rate",true);

                            _camera->setNodeMask(0xffffffff);
                            _switch->setValue(_frameRateChildNum, true);
                            break;
                        }
                        case(VIEWER_STATS):
                        {
                            ViewerBase::Scenes scenes;
                            viewer->getScenes(scenes);
                            for(ViewerBase::Scenes::iterator itr = scenes.begin();
                                itr != scenes.end();
                                ++itr)
                            {
                                Scene* scene = *itr;
                                osgDB::DatabasePager* dp = scene->getDatabasePager();
                                if (dp && dp->isRunning())
                                {
                                    dp->resetStats();
                                }
                            }

                            //viewer->getViewerStats()->collectStats("event",true);
                            //viewer->getViewerStats()->collectStats("update",true);

                            for(osgViewer::ViewerBase::Cameras::iterator itr = cameras.begin();
                                itr != cameras.end();
                                ++itr)
                            {
                                //if ((*itr)->getStats()) (*itr)->getStats()->collectStats("rendering",true);
                                //if ((*itr)->getStats()) (*itr)->getStats()->collectStats("gpu",true);
				if ((*itr)->getStats()) (*itr)->getStats()->collectStats("mgpu",true);
                            }

                            _camera->setNodeMask(0xffffffff);
                            _switch->setValue(_viewerChildNum, true);
                            break;
                        }
                        case(CAMERA_SCENE_STATS):
                        {
                            _camera->setNodeMask(0xffffffff);
                            _switch->setValue(_cameraSceneChildNum, true);

                            for(osgViewer::ViewerBase::Cameras::iterator itr = cameras.begin();
                                itr != cameras.end();
                                ++itr)
                            {
                                osg::Stats* stats = (*itr)->getStats();
                                if (stats)
                                {
                                    //stats->collectStats("scene",true);
                                }
                            }

                            break;
                        }
                        case(VIEWER_SCENE_STATS):
                        {
                            _camera->setNodeMask(0xffffffff);
                            _switch->setValue(_viewerSceneChildNum, true);

                            viewer->getViewerStats()->collectStats("scene",true);

                            break;
                        }
                        default:
                            break;
                    }

                    aa.requestRedraw();
                }
                return true;
            }
            if (ea.getKey()==_keyEventPrintsOutStats)
            {
                if (viewer->getViewerStats())
                {
                    osg::notify(osg::NOTICE)<<std::endl<<"Stats report:"<<std::endl;
                    typedef std::vector<osg::Stats*> StatsList;
                    StatsList statsList;
                    statsList.push_back(viewer->getViewerStats());

                    osgViewer::ViewerBase::Contexts contexts;
                    viewer->getContexts(contexts);
                    for(osgViewer::ViewerBase::Contexts::iterator gcitr = contexts.begin();
                        gcitr != contexts.end();
                        ++gcitr)
                    {
                        osg::GraphicsContext::Cameras& cameras = (*gcitr)->getCameras();
                        for(osg::GraphicsContext::Cameras::iterator itr = cameras.begin();
                            itr != cameras.end();
                            ++itr)
                        {
                            if ((*itr)->getStats())
                            {
                                statsList.push_back((*itr)->getStats());
                            }
                        }
                    }

                    for(int i = viewer->getViewerStats()->getEarliestFrameNumber(); i<= viewer->getViewerStats()->getLatestFrameNumber()-1; ++i)
                    {
                        for(StatsList::iterator itr = statsList.begin();
                            itr != statsList.end();
                            ++itr)
                        {
                            if (itr==statsList.begin()) (*itr)->report(osg::notify(osg::NOTICE), i);
                            else (*itr)->report(osg::notify(osg::NOTICE), i, "    ");
                        }
                        osg::notify(osg::NOTICE)<<std::endl;
                    }

                }
                return true;
            }
        }
        default: break;
    }

    return false;

}

void CustomStatsHandler::setUpScene(osgViewer::ViewerBase* viewer)
{
    _switch = new osg::Switch;

    _camera->addChild(_switch.get());

    osg::StateSet* stateset = _switch->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    stateset->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);
    stateset->setAttribute(new osg::PolygonMode(), osg::StateAttribute::PROTECTED);

    std::string font("fonts/arial.ttf");


    // collect all the relevant cameras
    ViewerBase::Cameras validCameras;
    viewer->getCameras(validCameras);

    ViewerBase::Cameras cameras;
    for(ViewerBase::Cameras::iterator itr = validCameras.begin();
        itr != validCameras.end();
        ++itr)
    {
        if ((*itr)->getStats())
        {
            cameras.push_back(*itr);
        }
    }

    // check for query time support
    unsigned int numCamrasWithTimerQuerySupport = 0;
    for(ViewerBase::Cameras::iterator citr = cameras.begin();
        citr != cameras.end();
        ++citr)
    {
        if ((*citr)->getGraphicsContext())
        {
            unsigned int contextID = (*citr)->getGraphicsContext()->getState()->getContextID();
            const osg::Drawable::Extensions* extensions = osg::Drawable::getExtensions(contextID, false);
            if (extensions && extensions->isTimerQuerySupported())
            {
                ++numCamrasWithTimerQuerySupport;
            }
        }
    }

    bool acquireGPUStats = numCamrasWithTimerQuerySupport==cameras.size();

    float leftPos = 10.0f;
    float startBlocks = 150.0f;
    float characterSize = 25.0f;

    osg::Vec3 pos(leftPos, _statsHeight-24.0f,0.0f);

    osg::Vec4 colorFR(1.0f,1.0f,1.0f,1.0f);
    osg::Vec4 colorFRAlpha(1.0f,1.0f,1.0f,0.5f);
    osg::Vec4 colorUpdate( 0.0f,1.0f,0.0f,1.0f);
    osg::Vec4 colorUpdateAlpha( 0.0f,1.0f,0.0f,0.5f);
    osg::Vec4 colorEvent(0.0f, 1.0f, 0.5f, 1.0f);
    osg::Vec4 colorEventAlpha(0.0f, 1.0f, 0.5f, 0.5f);
    osg::Vec4 colorCull( 0.0f,1.0f,1.0f,1.0f);
    osg::Vec4 colorCullAlpha( 0.0f,1.0f,1.0f,0.5f);
    osg::Vec4 colorDraw( 1.0f,1.0f,0.0f,1.0f);
    osg::Vec4 colorDrawAlpha( 1.0f,1.0f,0.0f,0.5f);
    osg::Vec4 colorGPU( 1.0f,0.5f,0.0f,1.0f);
    osg::Vec4 colorGPUAlpha( 1.0f,0.5f,0.0f,0.5f);

    osg::Vec4 colorDP( 1.0f,1.0f,0.5f,1.0f);


    // frame rate stats
    {
        osg::Geode* geode = new osg::Geode();
        _frameRateChildNum = _switch->getNumChildren();
        _switch->addChild(geode, false);

        osg::ref_ptr<osgText::Text> frameRateLabel = new osgText::Text;
        geode->addDrawable( frameRateLabel.get() );

        frameRateLabel->setColor(colorFR);
        frameRateLabel->setFont(font);
        frameRateLabel->setCharacterSize(characterSize);
        frameRateLabel->setPosition(pos);
        frameRateLabel->setText("Frame Rate: ");

        pos.x() = frameRateLabel->getBound().xMax();

        osg::ref_ptr<osgText::Text> frameRateValue = new osgText::Text;
        geode->addDrawable( frameRateValue.get() );

        frameRateValue->setColor(colorFR);
        frameRateValue->setFont(font);
        frameRateValue->setCharacterSize(characterSize);
        frameRateValue->setPosition(pos);
        frameRateValue->setText("0.0");

        frameRateValue->setDrawCallback(new AveragedValueTextDrawCallback(viewer->getViewerStats(),"Frame rate",-1, true, 1.0));

        pos.y() -= characterSize*1.5f;

    }

    osg::Vec4 backgroundColor(0.0, 0.0, 0.0f, 0.3);
    osg::Vec4 staticTextColor(1.0, 1.0, 0.0f, 1.0);
    osg::Vec4 dynamicTextColor(1.0, 1.0, 1.0f, 1.0);
    float backgroundMargin = 5;
    float backgroundSpacing = 3;


    // viewer stats
    {
        osg::Group* group = new osg::Group;
        _viewerChildNum = _switch->getNumChildren();
        _switch->addChild(group, false);

        osg::Geode* geode = new osg::Geode();
        group->addChild(geode);


        {
            pos.x() = leftPos;

            _threadingModelText = new osgText::Text;
            geode->addDrawable( _threadingModelText.get() );

            _threadingModelText->setColor(colorFR);
            _threadingModelText->setFont(font);
            _threadingModelText->setCharacterSize(characterSize);
            _threadingModelText->setPosition(pos);

            updateThreadingModelText();

            pos.y() -= characterSize*1.5f;
        }

        float topOfViewerStats = pos.y() + characterSize;

        geode->addDrawable(createBackgroundRectangle(
            pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
            _statsWidth - 2 * backgroundMargin,
            (3 + 4.5 * cameras.size()) * characterSize + 2 * backgroundMargin,
            backgroundColor) );

        /*{
            pos.x() = leftPos;

            osg::ref_ptr<osgText::Text> eventLabel = new osgText::Text;
            geode->addDrawable( eventLabel.get() );

            eventLabel->setColor(colorUpdate);
            eventLabel->setFont(font);
            eventLabel->setCharacterSize(characterSize);
            eventLabel->setPosition(pos);
            eventLabel->setText("Event: ");

            pos.x() = eventLabel->getBound().xMax();

            osg::ref_ptr<osgText::Text> eventValue = new osgText::Text;
            geode->addDrawable( eventValue.get() );

            eventValue->setColor(colorUpdate);
            eventValue->setFont(font);
            eventValue->setCharacterSize(characterSize);
            eventValue->setPosition(pos);
            eventValue->setText("0.0");

            eventValue->setDrawCallback(new AveragedValueTextDrawCallback(viewer->getViewerStats(),"Event traversal time taken",-1, false, 1000.0));

            pos.x() = startBlocks;
            osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorUpdateAlpha, _numBlocks);
            geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewer->getViewerStats(), viewer->getViewerStats(), "Event traversal begin time", "Event traversal end time", -1, _numBlocks));
            geode->addDrawable(geometry);

            pos.y() -= characterSize*1.5f;
        }*/

        /*{
            pos.x() = leftPos;

            osg::ref_ptr<osgText::Text> updateLabel = new osgText::Text;
            geode->addDrawable( updateLabel.get() );

            updateLabel->setColor(colorUpdate);
            updateLabel->setFont(font);
            updateLabel->setCharacterSize(characterSize);
            updateLabel->setPosition(pos);
            updateLabel->setText("Update: ");

            pos.x() = updateLabel->getBound().xMax();

            osg::ref_ptr<osgText::Text> updateValue = new osgText::Text;
            geode->addDrawable( updateValue.get() );

            updateValue->setColor(colorUpdate);
            updateValue->setFont(font);
            updateValue->setCharacterSize(characterSize);
            updateValue->setPosition(pos);
            updateValue->setText("0.0");

            updateValue->setDrawCallback(new AveragedValueTextDrawCallback(viewer->getViewerStats(),"Update traversal time taken",-1, false, 1000.0));

            pos.x() = startBlocks;
            osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorUpdateAlpha, _numBlocks);
            geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewer->getViewerStats(), viewer->getViewerStats(), "Update traversal begin time", "Update traversal end time", -1, _numBlocks));
            geode->addDrawable(geometry);

            pos.y() -= characterSize*1.5f;
        }*/

        pos.x() = leftPos;

        // add camera stats
        for(ViewerBase::Cameras::iterator citr = cameras.begin();
            citr != cameras.end();
            ++citr)
        {
            group->addChild(createCameraTimeStats(font, pos, startBlocks, acquireGPUStats, characterSize, viewer->getViewerStats(), *citr));
        }

        // add frame ticks
        {
            osg::Geode* geode = new osg::Geode;
            group->addChild(geode);

            osg::Vec4 colourTicks(1.0f,1.0f,1.0f, 0.5f);

            pos.x() = startBlocks;
            pos.y() += characterSize;
            float height = topOfViewerStats - pos.y();

            osg::Geometry* ticks = createTick(pos, 5.0f, colourTicks, 100);
            geode->addDrawable(ticks);

            osg::Geometry* frameMarkers = createFrameMarkers(pos, height, colourTicks, _numBlocks + 1);
            frameMarkers->setDrawCallback(new FrameMarkerDrawCallback(this, startBlocks, viewer->getViewerStats(), 0, _numBlocks + 1));
            geode->addDrawable(frameMarkers);

            pos.x() = leftPos;
        }

        // Stats line graph
        {
            pos.y() -= (backgroundSpacing + 2 * backgroundMargin);
            float width = _statsWidth - 4 * backgroundMargin;
            float height = 5 * characterSize;

            // Create a stats graph and add any stats we want to track with it.
            StatsGraph* statsGraph = new StatsGraph(pos, width, height);
            group->addChild(statsGraph);

            statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorFR, 100, "Frame rate");
            //statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorEvent, 0.016, "Event traversal time taken");
            //statsGraph->addStatGraph(viewer->getViewerStats(), viewer->getViewerStats(), colorUpdate, 0.016, "Update traversal time taken");

            for(ViewerBase::Cameras::iterator citr = cameras.begin();
                citr != cameras.end();
                ++citr)
            {
                //statsGraph->addStatGraph(viewer->getViewerStats(), (*citr)->getStats(), colorCull, 0.016, "Cull traversal time taken");
                statsGraph->addStatGraph(viewer->getViewerStats(), (*citr)->getStats(), colorDraw, 0.016, "MDraw traversal time taken");
                //statsGraph->addStatGraph(viewer->getViewerStats(), (*citr)->getStats(), colorGPU, 0.016, "GPU draw time taken");
            }

            geode->addDrawable(createBackgroundRectangle( pos + osg::Vec3(-backgroundMargin, backgroundMargin, 0),
                                                          width + 2 * backgroundMargin,
                                                          height + 2 * backgroundMargin,
                                                          backgroundColor) );

            pos.x() = leftPos;
            pos.y() -= height + 2 * backgroundMargin;
        }

        // Databasepager stats
        ViewerBase::Scenes scenes;
        viewer->getScenes(scenes);
        for(ViewerBase::Scenes::iterator itr = scenes.begin();
            itr != scenes.end();
            ++itr)
        {
            Scene* scene = *itr;
            osgDB::DatabasePager* dp = scene->getDatabasePager();
            if (dp && dp->isRunning())
            {
                pos.y() -= (characterSize + backgroundSpacing);

                geode->addDrawable(createBackgroundRectangle(    pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
                                                                _statsWidth - 2 * backgroundMargin,
                                                                characterSize + 2 * backgroundMargin,
                                                                backgroundColor));

                osg::ref_ptr<osgText::Text> averageLabel = new osgText::Text;
                geode->addDrawable( averageLabel.get() );

                averageLabel->setColor(colorDP);
                averageLabel->setFont(font);
                averageLabel->setCharacterSize(characterSize);
                averageLabel->setPosition(pos);
                averageLabel->setText("DatabasePager time to merge new tiles - average: ");

                pos.x() = averageLabel->getBound().xMax();

                osg::ref_ptr<osgText::Text> averageValue = new osgText::Text;
                geode->addDrawable( averageValue.get() );

                averageValue->setColor(colorDP);
                averageValue->setFont(font);
                averageValue->setCharacterSize(characterSize);
                averageValue->setPosition(pos);
                averageValue->setText("1000");

                pos.x() = averageValue->getBound().xMax() + 2.0f*characterSize;


                osg::ref_ptr<osgText::Text> minLabel = new osgText::Text;
                geode->addDrawable( minLabel.get() );

                minLabel->setColor(colorDP);
                minLabel->setFont(font);
                minLabel->setCharacterSize(characterSize);
                minLabel->setPosition(pos);
                minLabel->setText("min: ");

                pos.x() = minLabel->getBound().xMax();

                osg::ref_ptr<osgText::Text> minValue = new osgText::Text;
                geode->addDrawable( minValue.get() );

                minValue->setColor(colorDP);
                minValue->setFont(font);
                minValue->setCharacterSize(characterSize);
                minValue->setPosition(pos);
                minValue->setText("1000");

                pos.x() = minValue->getBound().xMax() + 2.0f*characterSize;

                osg::ref_ptr<osgText::Text> maxLabel = new osgText::Text;
                geode->addDrawable( maxLabel.get() );

                maxLabel->setColor(colorDP);
                maxLabel->setFont(font);
                maxLabel->setCharacterSize(characterSize);
                maxLabel->setPosition(pos);
                maxLabel->setText("max: ");

                pos.x() = maxLabel->getBound().xMax();

                osg::ref_ptr<osgText::Text> maxValue = new osgText::Text;
                geode->addDrawable( maxValue.get() );

                maxValue->setColor(colorDP);
                maxValue->setFont(font);
                maxValue->setCharacterSize(characterSize);
                maxValue->setPosition(pos);
                maxValue->setText("1000");

                pos.x() = maxValue->getBound().xMax();

                osg::ref_ptr<osgText::Text> requestsLabel = new osgText::Text;
                geode->addDrawable( requestsLabel.get() );

                requestsLabel->setColor(colorDP);
                requestsLabel->setFont(font);
                requestsLabel->setCharacterSize(characterSize);
                requestsLabel->setPosition(pos);
                requestsLabel->setText("requests: ");

                pos.x() = requestsLabel->getBound().xMax();

                osg::ref_ptr<osgText::Text> requestList = new osgText::Text;
                geode->addDrawable( requestList.get() );

                requestList->setColor(colorDP);
                requestList->setFont(font);
                requestList->setCharacterSize(characterSize);
                requestList->setPosition(pos);
                requestList->setText("0");

                pos.x() = requestList->getBound().xMax() + 2.0f*characterSize;;

                osg::ref_ptr<osgText::Text> compileLabel = new osgText::Text;
                geode->addDrawable( compileLabel.get() );

                compileLabel->setColor(colorDP);
                compileLabel->setFont(font);
                compileLabel->setCharacterSize(characterSize);
                compileLabel->setPosition(pos);
                compileLabel->setText("tocompile: ");

                pos.x() = compileLabel->getBound().xMax();

                osg::ref_ptr<osgText::Text> compileList = new osgText::Text;
                geode->addDrawable( compileList.get() );

                compileList->setColor(colorDP);
                compileList->setFont(font);
                compileList->setCharacterSize(characterSize);
                compileList->setPosition(pos);
                compileList->setText("0");

                pos.x() = maxLabel->getBound().xMax();

                geode->setCullCallback(new PagerCallback(dp, minValue.get(), maxValue.get(), averageValue.get(), requestList.get(), compileList.get(), 1000.0));
            }

            pos.x() = leftPos;
        }
    }

    // Camera scene stats
    {
        pos.y() -= (characterSize + backgroundSpacing + 2 * backgroundMargin);

        osg::Group* group = new osg::Group;
        _cameraSceneChildNum = _switch->getNumChildren();
        _switch->addChild(group, false);

        osg::Geode* geode = new osg::Geode();
        geode->setCullingActive(false);
        group->addChild(geode);
        geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
                                                        7 * characterSize + 2 * backgroundMargin,
                                                        19 * characterSize + 2 * backgroundMargin,
                                                        backgroundColor));

        // Camera scene & primitive stats static text
        osg::ref_ptr<osgText::Text> camStaticText = new osgText::Text;
        geode->addDrawable( camStaticText.get() );
        camStaticText->setColor(staticTextColor);
        camStaticText->setFont(font);
        camStaticText->setCharacterSize(characterSize);
        camStaticText->setPosition(pos);

        std::ostringstream viewStr;
        viewStr.clear();
        viewStr.setf(std::ios::left, std::ios::adjustfield);
        viewStr.width(14);
        viewStr << "Camera" << std::endl;
        viewStr << "" << std::endl; // placeholder for Camera name
        viewStr << "Lights" << std::endl;
        viewStr << "Bins" << std::endl;
        viewStr << "Depth" << std::endl;
        viewStr << "Materials" << std::endl;
        viewStr << "Imposters" << std::endl;
        viewStr << "Drawables" << std::endl;
        viewStr << "Vertices" << std::endl;
        viewStr << "Points" << std::endl;
        viewStr << "Lines" << std::endl;
        viewStr << "Line strips" << std::endl;
        viewStr << "Line loops" << std::endl;
        viewStr << "Triangles" << std::endl;
        viewStr << "Tri. strips" << std::endl;
        viewStr << "Tri. fans" << std::endl;
        viewStr << "Quads" << std::endl;
        viewStr << "Quad strips" << std::endl;
        viewStr << "Polygons" << std::endl;
        viewStr.setf(std::ios::right,std::ios::adjustfield);
        camStaticText->setText(viewStr.str());

        // Move camera block to the right
        pos.x() += 7 * characterSize + 2 * backgroundMargin + backgroundSpacing;

        // Add camera scene stats, one block per camera
        int cameraCounter = 0;
        for(ViewerBase::Cameras::iterator citr = cameras.begin(); citr != cameras.end(); ++citr)
        {
            geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
                                                            5 * characterSize + 2 * backgroundMargin,
                                                            19 * characterSize + 2 * backgroundMargin,
                                                            backgroundColor));

            // Camera scene stats
            osg::ref_ptr<osgText::Text> camStatsText = new osgText::Text;
            geode->addDrawable( camStatsText.get() );

            camStatsText->setColor(dynamicTextColor);
            camStatsText->setFont(font);
            camStatsText->setCharacterSize(characterSize);
            camStatsText->setPosition(pos);
            camStatsText->setText("");
            camStatsText->setDrawCallback(new CameraSceneStatsTextDrawCallback(*citr, cameraCounter));

            // Move camera block to the right
            pos.x() +=  5 * characterSize + 2 * backgroundMargin + backgroundSpacing;
            cameraCounter++;
        }
    }

    // Viewer scene stats
    {
        osg::Group* group = new osg::Group;
        _viewerSceneChildNum = _switch->getNumChildren();
        _switch->addChild(group, false);

        osg::Geode* geode = new osg::Geode();
        geode->setCullingActive(false);
        group->addChild(geode);

        geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
                                                        6 * characterSize + 2 * backgroundMargin,
                                                        12 * characterSize + 2 * backgroundMargin,
                                                        backgroundColor));

        // View scene stats static text
        osg::ref_ptr<osgText::Text> camStaticText = new osgText::Text;
        geode->addDrawable( camStaticText.get() );
        camStaticText->setColor(staticTextColor);
        camStaticText->setFont(font);
        camStaticText->setCharacterSize(characterSize);
        camStaticText->setPosition(pos);

        std::ostringstream viewStr;
        viewStr.clear();
        viewStr.setf(std::ios::left, std::ios::adjustfield);
        viewStr.width(14);
        viewStr << "View" << std::endl;
        viewStr << " " << std::endl;
        viewStr << "Stateset" << std::endl;
        viewStr << "Group" << std::endl;
        viewStr << "Transform" << std::endl;
        viewStr << "LOD" << std::endl;
        viewStr << "Switch" << std::endl;
        viewStr << "Geode" << std::endl;
        viewStr << "Drawable" << std::endl;
        viewStr << "Geometry" << std::endl;
        viewStr << "Vertices" << std::endl;
        viewStr << "Primitives" << std::endl;
        viewStr.setf(std::ios::right, std::ios::adjustfield);
        camStaticText->setText(viewStr.str());

        // Move viewer block to the right
        pos.x() += 6 * characterSize + 2 * backgroundMargin + backgroundSpacing;

        std::vector<osgViewer::View*> views;
        viewer->getViews(views);

        std::vector<osgViewer::View*>::iterator it;
        int viewCounter = 0;
        for (it = views.begin(); it != views.end(); ++it)
        {
            geode->addDrawable(createBackgroundRectangle(pos + osg::Vec3(-backgroundMargin, characterSize + backgroundMargin, 0),
                                                            10 * characterSize + 2 * backgroundMargin,
                                                            12 * characterSize + 2 * backgroundMargin,
                                                            backgroundColor));

            // Text for scene statistics
            osgText::Text* text = new  osgText::Text;
            geode->addDrawable( text );

            text->setColor(dynamicTextColor);
            text->setFont(font);
            text->setCharacterSize(characterSize);
            text->setPosition(pos);
            text->setDrawCallback(new ViewSceneStatsTextDrawCallback(*it, viewCounter));

            pos.x() += 10 * characterSize + 2 * backgroundMargin + backgroundSpacing;
            viewCounter++;
        }
    }
}

osg::Node* CustomStatsHandler::createCameraTimeStats(const std::string& font, osg::Vec3& pos, float startBlocks, bool acquireGPUStats, float characterSize, osg::Stats* viewerStats, osg::Camera* camera)
{
    osg::Stats* stats = camera->getStats();
    if (!stats) return 0;

    int gpu = camera->getGraphicsContext()->getTraits()->screenNum;
    //std::cerr << "Making time stats for camera " << gpu << std::endl;

    osg::Group* group = new osg::Group;

    osg::Geode* geode = new osg::Geode();
    group->addChild(geode);

    float leftPos = pos.x();

    if(gpu >= _numGPUs)
    {
	return group;
    }

    osg::Vec4 colorCull( 0.0f,1.0f,1.0f,1.0f);
    osg::Vec4 colorCullAlpha( 0.0f,1.0f,1.0f,0.5f);
    osg::Vec4 colorDraw( 1.0f,1.0f,0.0f,1.0f);
    osg::Vec4 colorDrawAlpha( 1.0f,1.0f,0.0f,0.5f);
    osg::Vec4 colorGPU( 1.0f,0.5f,0.0f,1.0f);
    osg::Vec4 colorGPUAlpha( 1.0f,0.5f,0.0f,0.5f);

    osg::Vec4 colorCopyDown(0.0,1.0,1.0,1.0);
    osg::Vec4 colorCopyDownAlpha(0.0,1.0,1.0,0.5);

    osg::Vec4 colorShader(1.0,0.5,0.0,1.0);
    osg::Vec4 colorShaderAlpha(1.0,0.5,0.0,0.5);

    osg::Vec4 colorSetFrame(1.0,0.0,1.0,1.0);
    osg::Vec4 colorSetFrameAlpha(1.0,0.0,1.0,0.5);

    osg::Vec4 colorMultCopy(0.0,0.0,1.0,1.0);
    osg::Vec4 colorMultCopyAlpha(0.0,0.0,1.0,0.5);

    osg::Vec4 colorPostChc(0.7,0.7,0.7,1.0);
    osg::Vec4 colorPostChcAlpha(0.7,0.7,0.7,0.5);

    osg::Vec4 colorCopyBack[_numGPUs];
    osg::Vec4 colorCopyBackAlpha[_numGPUs];
    colorCopyBack[1] = osg::Vec4(1.0,0.0,0.0,1.0);
    colorCopyBackAlpha[1] = osg::Vec4(1.0,0.0,0.0,0.5);
    colorCopyBack[2] = osg::Vec4(0.0,1.0,0.0,1.0);
    colorCopyBackAlpha[2] = osg::Vec4(0.0,1.0,0.0,0.5);
    colorCopyBack[3] = osg::Vec4(0.0,0.0,1.0,1.0);
    colorCopyBackAlpha[3] = osg::Vec4(0.0,0.0,1.0,0.5);

    {
        pos.x() = leftPos;

        osg::ref_ptr<osgText::Text> drawLabel = new osgText::Text;
        geode->addDrawable( drawLabel.get() );

        drawLabel->setColor(colorSetFrame);
        drawLabel->setFont(font);
        drawLabel->setCharacterSize(characterSize);
        drawLabel->setPosition(pos);
        drawLabel->setText("Set Frame: ");

        pos.x() = drawLabel->getBound().xMax();

        osg::ref_ptr<osgText::Text> drawValue = new osgText::Text;
        geode->addDrawable( drawValue.get() );

        drawValue->setColor(colorSetFrame);
        drawValue->setFont(font);
        drawValue->setCharacterSize(characterSize);
        drawValue->setPosition(pos);
        drawValue->setText("0.0");

        drawValue->setDrawCallback(new AveragedValueTextDrawCallback(stats,"SetFrame time taken",-1, false, 1000.0));


        pos.x() = startBlocks;
        osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorSetFrameAlpha, _numBlocks);
        geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, "SetFrame begin time", "SetFrame end time", -1, _numBlocks));
        geode->addDrawable(geometry);

        pos.y() -= characterSize*1.5f;
    }

    {
        pos.x() = leftPos;

        osg::ref_ptr<osgText::Text> drawLabel = new osgText::Text;
        geode->addDrawable( drawLabel.get() );

        drawLabel->setColor(colorMultCopy);
        drawLabel->setFont(font);
        drawLabel->setCharacterSize(characterSize);
        drawLabel->setPosition(pos);
        drawLabel->setText("Data Copy: ");

        pos.x() = drawLabel->getBound().xMax();

        osg::ref_ptr<osgText::Text> drawValue = new osgText::Text;
        geode->addDrawable( drawValue.get() );

        drawValue->setColor(colorMultCopy);
        drawValue->setFont(font);
        drawValue->setCharacterSize(characterSize);
        drawValue->setPosition(pos);
        drawValue->setText("0.0");

        drawValue->setDrawCallback(new AveragedValueTextDrawCallback(stats,"LoadFrameData time taken",-1, false, 1000.0));


        pos.x() = startBlocks;
        osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorMultCopyAlpha, _numBlocks);
        geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, "LoadFrameData begin time", "LoadFrameData end time", -1, _numBlocks));
        geode->addDrawable(geometry);

        pos.y() -= characterSize*1.5f;
    }

    {
        pos.x() = leftPos;

        osg::ref_ptr<osgText::Text> drawLabel = new osgText::Text;
        geode->addDrawable( drawLabel.get() );

        drawLabel->setColor(colorDraw);
        drawLabel->setFont(font);
        drawLabel->setCharacterSize(characterSize);
        drawLabel->setPosition(pos);
        drawLabel->setText("Draw: ");

        pos.x() = drawLabel->getBound().xMax();

        osg::ref_ptr<osgText::Text> drawValue = new osgText::Text;
        geode->addDrawable( drawValue.get() );

        drawValue->setColor(colorDraw);
        drawValue->setFont(font);
        drawValue->setCharacterSize(characterSize);
        drawValue->setPosition(pos);
        drawValue->setText("0.0");

        drawValue->setDrawCallback(new AveragedValueTextDrawCallback(stats,"MDraw traversal time taken",-1, false, 1000.0));


        pos.x() = startBlocks;
        osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorDrawAlpha, _numBlocks);
        geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, "MDraw traversal begin time", "MDraw traversal end time", -1, _numBlocks));
        geode->addDrawable(geometry);

        pos.y() -= characterSize*1.5f;
    }

    if(gpu == 0)
    {
	for(int i = 1; i < _numGPUs; i++)
	{
	    pos.x() = leftPos;

	    osg::ref_ptr<osgText::Text> Label = new osgText::Text;
	    geode->addDrawable( Label.get() );

	    std::stringstream text;
	    text << "CopyBack " << i << ": ";

	    Label->setColor(colorCopyBack[i]);
	    Label->setFont(font);
	    Label->setCharacterSize(characterSize);
	    Label->setPosition(pos);
	    Label->setText(text.str());

	    pos.x() = Label->getBound().xMax();

	    osg::ref_ptr<osgText::Text> Value = new osgText::Text;
	    geode->addDrawable( Value.get() );

	    Value->setColor(colorCopyBack[i]);
	    Value->setFont(font);
	    Value->setCharacterSize(characterSize);
	    Value->setPosition(pos);
	    Value->setText("0.0");

	    std::stringstream ss;
	    ss << "CopyBack" << i;
	    Value->setDrawCallback(new AveragedValueTextDrawCallback(stats, ss.str() + " time taken",-1, false, 1000.0));


	    pos.x() = startBlocks;
	    osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorCopyBackAlpha[i], _numBlocks);
	    geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, ss.str() + " begin time", ss.str() + " end time", -1, _numBlocks));
	    geode->addDrawable(geometry);

	    pos.y() -= characterSize*1.5f;
	}

	{
	    pos.x() = leftPos;

	    osg::ref_ptr<osgText::Text> Label = new osgText::Text;
	    geode->addDrawable( Label.get() );

	    Label->setColor(colorShader);
	    Label->setFont(font);
	    Label->setCharacterSize(characterSize);
	    Label->setPosition(pos);
	    Label->setText("Shader: ");

	    pos.x() = Label->getBound().xMax();

	    osg::ref_ptr<osgText::Text> Value = new osgText::Text;
	    geode->addDrawable( Value.get() );

	    Value->setColor(colorShader);
	    Value->setFont(font);
	    Value->setCharacterSize(characterSize);
	    Value->setPosition(pos);
	    Value->setText("0.0");

	    Value->setDrawCallback(new AveragedValueTextDrawCallback(stats,"Shader time taken",-1, false, 1000.0));


	    pos.x() = startBlocks;
	    osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorShaderAlpha, _numBlocks);
	    geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, "Shader begin time", "Shader end time", -1, _numBlocks));
	    geode->addDrawable(geometry);

	    pos.y() -= characterSize*1.5f;
	}

	{
	    pos.x() = leftPos;

	    osg::ref_ptr<osgText::Text> Label = new osgText::Text;
	    geode->addDrawable( Label.get() );

	    Label->setColor(colorPostChc);
	    Label->setFont(font);
	    Label->setCharacterSize(characterSize);
	    Label->setPosition(pos);
	    Label->setText("Post CHC: ");

	    pos.x() = Label->getBound().xMax();

	    osg::ref_ptr<osgText::Text> Value = new osgText::Text;
	    geode->addDrawable( Value.get() );

	    Value->setColor(colorPostChc);
	    Value->setFont(font);
	    Value->setCharacterSize(characterSize);
	    Value->setPosition(pos);
	    Value->setText("0.0");

	    Value->setDrawCallback(new AveragedValueTextDrawCallback(stats,"PostCHC time taken",-1, false, 1000.0));


	    pos.x() = startBlocks;
	    osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorPostChcAlpha, _numBlocks);
	    geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, "PostCHC begin time", "PostCHC end time", -1, _numBlocks));
	    geode->addDrawable(geometry);

	    pos.y() -= characterSize*1.5f;
	}
    }
    else
    {
	pos.x() = leftPos;

	osg::ref_ptr<osgText::Text> Label = new osgText::Text;
	geode->addDrawable( Label.get() );

	std::stringstream text;
	text << "CopyDown " << gpu << ": ";

	Label->setColor(colorCopyDown);
	Label->setFont(font);
	Label->setCharacterSize(characterSize);
	Label->setPosition(pos);
	Label->setText(text.str());

	pos.x() = Label->getBound().xMax();

	osg::ref_ptr<osgText::Text> Value = new osgText::Text;
	geode->addDrawable( Value.get() );

	Value->setColor(colorCopyDown);
	Value->setFont(font);
	Value->setCharacterSize(characterSize);
	Value->setPosition(pos);
	Value->setText("0.0");

	std::stringstream ss;
	ss << "CopyDown" << gpu;
	Value->setDrawCallback(new AveragedValueTextDrawCallback(stats, ss.str() + " time taken",-1, false, 1000.0));


	pos.x() = startBlocks;
	osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorCopyDownAlpha, _numBlocks);
	geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, ss.str() + " begin time", ss.str() + " end time", -1, _numBlocks));
	geode->addDrawable(geometry);

	pos.y() -= characterSize*1.5f;
    }

    /*{
        pos.x() = leftPos;

        osg::ref_ptr<osgText::Text> cullLabel = new osgText::Text;
        geode->addDrawable( cullLabel.get() );

        cullLabel->setColor(colorCull);
        cullLabel->setFont(font);
        cullLabel->setCharacterSize(characterSize);
        cullLabel->setPosition(pos);
        cullLabel->setText("Cull: ");

        pos.x() = cullLabel->getBound().xMax();

        osg::ref_ptr<osgText::Text> cullValue = new osgText::Text;
        geode->addDrawable( cullValue.get() );

        cullValue->setColor(colorCull);
        cullValue->setFont(font);
        cullValue->setCharacterSize(characterSize);
        cullValue->setPosition(pos);
        cullValue->setText("0.0");

        cullValue->setDrawCallback(new AveragedValueTextDrawCallback(stats,"Cull traversal time taken",-1, false, 1000.0));

        pos.x() = startBlocks;
        osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorCullAlpha, _numBlocks);
        geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, "Cull traversal begin time", "Cull traversal end time", -1, _numBlocks));
        geode->addDrawable(geometry);

        pos.y() -= characterSize*1.5f;
    }

    {
        pos.x() = leftPos;

        osg::ref_ptr<osgText::Text> drawLabel = new osgText::Text;
        geode->addDrawable( drawLabel.get() );

        drawLabel->setColor(colorDraw);
        drawLabel->setFont(font);
        drawLabel->setCharacterSize(characterSize);
        drawLabel->setPosition(pos);
        drawLabel->setText("Draw: ");

        pos.x() = drawLabel->getBound().xMax();

        osg::ref_ptr<osgText::Text> drawValue = new osgText::Text;
        geode->addDrawable( drawValue.get() );

        drawValue->setColor(colorDraw);
        drawValue->setFont(font);
        drawValue->setCharacterSize(characterSize);
        drawValue->setPosition(pos);
        drawValue->setText("0.0");

        drawValue->setDrawCallback(new AveragedValueTextDrawCallback(stats,"Draw traversal time taken",-1, false, 1000.0));


        pos.x() = startBlocks;
        osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorDrawAlpha, _numBlocks);
        geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, "Draw traversal begin time", "Draw traversal end time", -1, _numBlocks));
        geode->addDrawable(geometry);

        pos.y() -= characterSize*1.5f;
    }

    if (acquireGPUStats)
    {
        pos.x() = leftPos;

        osg::ref_ptr<osgText::Text> gpuLabel = new osgText::Text;
        geode->addDrawable( gpuLabel.get() );

        gpuLabel->setColor(colorGPU);
        gpuLabel->setFont(font);
        gpuLabel->setCharacterSize(characterSize);
        gpuLabel->setPosition(pos);
        gpuLabel->setText("GPU: ");

        pos.x() = gpuLabel->getBound().xMax();

        osg::ref_ptr<osgText::Text> gpuValue = new osgText::Text;
        geode->addDrawable( gpuValue.get() );

        gpuValue->setColor(colorGPU);
        gpuValue->setFont(font);
        gpuValue->setCharacterSize(characterSize);
        gpuValue->setPosition(pos);
        gpuValue->setText("0.0");

        gpuValue->setDrawCallback(new AveragedValueTextDrawCallback(stats,"GPU draw time taken",-1, false, 1000.0));

        pos.x() = startBlocks;
        osg::Geometry* geometry = createGeometry(pos, characterSize *0.8, colorGPUAlpha, _numBlocks);
        geometry->setDrawCallback(new BlockDrawCallback(this, startBlocks, viewerStats, stats, "GPU draw begin time", "GPU draw end time", -1, _numBlocks));
        geode->addDrawable(geometry);

        pos.y() -= characterSize*1.5f;
    }*/


    pos.x() = leftPos;

    return group;
}
