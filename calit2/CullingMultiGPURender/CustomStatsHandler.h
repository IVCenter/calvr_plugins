/**
 * @file CustomStatsHandler.h
 * Contains class to display custom timings for our algorithm 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#ifndef CUSTOM_STATS_HANDLER_H
#define CUSTOM_STATS_HANDLER_H

#include <osgViewer/ViewerEventHandlers>

/**
 * Class to handle display of custom timing data for our algorithm
 */
class CustomStatsHandler : public osgViewer::StatsHandler
{
    public:
        /**
         * @param gpus number of gpus used for parallel draw
         */
        CustomStatsHandler(int gpus);
        virtual ~CustomStatsHandler();

        /// event handler function
        virtual bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa);

    protected:
        /// setup stats
        void setUpScene(osgViewer::ViewerBase* viewer);
        /// setup stats per camera
        osg::Node* createCameraTimeStats(const std::string& font, osg::Vec3& pos, float startBlocks, bool acquireGPUStats, float characterSize, osg::Stats* viewerStats, osg::Camera* camera);

        int _numGPUs;   ///< number of gpus used for parallel draw
};

#endif
