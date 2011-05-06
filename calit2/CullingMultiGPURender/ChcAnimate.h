#ifndef CHCANIMATE_H
#define CHCANIMATE_H

#include "MultiGPURenderer.h"
#include "ChcPreDrawCallback.h"
#include "RenderTraverser.h"
#include "HierarchyNode.h"
#include "FetchQueue.h"
#include "Geometry.h"
#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <OpenThreads/Mutex>

/** ChcAnimate class deals with controlling the animation, determining
 *  the next frame to render.
 */
class ChcAnimate
{
public:
	ChcAnimate(std::string filename, int numcontexts = 1);
        /// Sets the next frame to render
        void setNextFrame();
        void postRender();
        void postRenderPerThread(int context);
        /// Update the eye view and projection so view frustrum and occulsion culling can occur.
        void updateViewParameters(double* eyeview, double* eyeprojview);

        std::map<int, Geometry* > * getGeometryMap();
        std::vector<int> * getPartList(int); 

        void setFrame(int frame);
        void advance();

        void play();
        void pause();

        bool getPaused();

        void update();

        int getFrame();
        int getNextFrame();

        void turnOffGeometry();

protected:

        ChcAnimate();
        ~ChcAnimate();
        void loadFrameMetaData(std::string filename);
        HierarchyNode* DecodeState(ifstream &in, HierarchyNode* parent, int level, int frameNum, std::string filename); 
        

        RenderTraverser *traverser;
        std::vector<HierarchyNode*> frameSetup;
        std::map<int,Geometry*> * pmap; ///< note Geometry* can be cast to PartInfo*
        std::vector< std::vector<int> * > plist;
        int currentFrame;
        int lastFrame;
        int numContexts;
	double eyePos[3];		///< vec3
	double eyeProjView[16];		///< matrix4x4

        bool _cudaCopy;
        bool _paused;
};
#endif // CHCANIMATE
