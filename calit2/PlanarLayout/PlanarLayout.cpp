#include "PlanarLayout.h"

#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginHelper.h>
#include <PluginMessageType.h>

#include <iostream>
#include <cmath>

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN(PlanarLayout)

bool PlanarLayout::init()
{
    cerr << "PlanarLayout init" << endl;

    if (SceneManager::instance()->getTiledWallValid())
    {
        mLayoutAlgorithm = new PlanarLayoutAlgorithm;

        LayoutManagerAddLayoutData lmald;
        lmald.layout = mLayoutAlgorithm;

        PluginHelper::sendMessageByName( "LayoutManager", LM_ADD_LAYOUT, (char*)&lmald);
    }

    cerr << "PlanarLayout init done.\n";

    return SceneManager::instance()->getTiledWallValid();
}


PlanarLayout::~PlanarLayout()
{
    delete mLayoutAlgorithm;
}

/*virtual*/ string
PlanarLayout::PlanarLayoutAlgorithm::Name( void )
{
    return "Planar Layout";
}

/*virtual*/ void
PlanarLayout::PlanarLayoutAlgorithm::Cleanup( void )
{
    mObjTrans.clear();
}

/*virtual*/ bool
PlanarLayout::PlanarLayoutAlgorithm::Start( void )
{
    mTimeElapsed = 0.0;
    mObjTrans.clear();

    vector< SceneObject* > movable_objects;

    {
        vector< SceneObject* > scene_objects = SceneManager::instance()->getSceneObjects();

        for (unsigned int i = 0; i < scene_objects.size(); ++i)
        {
            if (scene_objects[i]->getMovable() && !scene_objects[i]->getNavigationOn())
                movable_objects.push_back( scene_objects[i] );
        }

        if (movable_objects.empty())
            return false;
    }

    unsigned int cols = (unsigned int)ceil( sqrt(movable_objects.size()) * SceneManager::instance()->getTiledWallWidth() / (double)SceneManager::instance()->getTiledWallHeight() );
    unsigned int rows = (unsigned int)ceil(movable_objects.size() / (double)cols);

    while (GetBestGrid(rows, cols, movable_objects.size()))
        ;

    double x_inc = SceneManager::instance()->getTiledWallWidth() / cols;
    double z_inc = SceneManager::instance()->getTiledWallHeight() / rows;

    unsigned int m = 0;

    for (unsigned int r = 0; r < rows && m < movable_objects.size(); ++r)
    {
        double pos_z = SceneManager::instance()->getTiledWallHeight() / 2.0
                    - (z_inc / 2.0) - (r * z_inc);

        for (unsigned int c = 0; c < cols && m < movable_objects.size(); ++c)
        {
            double pos_x = SceneManager::instance()->getTiledWallWidth() / -2.0
                        + (x_inc / 2.0) + (c * x_inc);

            TransitionState trans;

            trans.startScale = movable_objects[m]->getScale();
            osg::BoundingBox box = movable_objects[m]->getOrComputeBoundingBox();
            double x_scale = x_inc / (box.xMax()-box.xMin());
            double z_scale = z_inc / (box.zMax()-box.zMin());
            trans.endScale = (x_scale < z_scale) ? x_scale : z_scale;

            trans.startPos = movable_objects[m]->getPosition();
            trans.endPos = Vec3(pos_x, 0, pos_z);
            trans.endPos -= box.center() * trans.endScale;
            trans.endPos = SceneManager::instance()->getTiledWallTransform() * trans.endPos;

            mObjTrans[ movable_objects[m] ] = trans;
            ++m;
        }
    }

    return true;
}

/*virtual*/ bool
PlanarLayout::PlanarLayoutAlgorithm::Update( void )
{
    double const TOTAL_TIME = 3.0;

    mTimeElapsed += PluginHelper::getLastFrameDuration();

    if (mTimeElapsed > TOTAL_TIME)
        mTimeElapsed = TOTAL_TIME;

    vector< SceneObject* > scene_objects = SceneManager::instance()->getSceneObjects();

    for (unsigned int i = 0; i < scene_objects.size(); ++i)
    {
        SceneObjTrans::iterator it = mObjTrans.find( scene_objects[i] );

        if (mObjTrans.end() == it)
            continue;

        double progress = 1.0 - mTimeElapsed / TOTAL_TIME;

        Vec3 pos = it->second.endPos + (it->second.startPos - it->second.endPos) * progress;
        float scale = it->second.endScale + (it->second.startScale - it->second.endScale) * progress;

        it->first->setPosition( pos );
        it->first->setScale( scale );
    }

    return (TOTAL_TIME <= mTimeElapsed);
}

bool
PlanarLayout::PlanarLayoutAlgorithm::GetBestGrid(unsigned int &r, unsigned int &c, unsigned int cells) const
{
    unsigned int best_r = r, best_c = c;
    double best_rating = Rate(r, c, cells);

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j<= 1; j++)
        {
            if (0 == i && 0 == j)
                continue;

            double rating = Rate(r+i,c+j, cells);
            if (rating > best_rating)
            {
                best_rating = rating;
                best_r = r+i;
                best_c = c+j;
            }
        }
    }

    if (best_r == r && best_c == c)
        return false;

    r = best_r;
    c = best_c;
    return true;
}

double
PlanarLayout::PlanarLayoutAlgorithm::Rate(unsigned int r, unsigned int c, unsigned int cells) const
{
    if (r*c < cells || r < 0 || c < 0)
        return 0;

    double ratio = (SceneManager::instance()->getTiledWallWidth() * r)
                   / (SceneManager::instance()->getTiledWallHeight() * c);
    if (ratio > 1)
        ratio = 1 / ratio;

    double mod = cells / (double)(r*c);
    mod = mod * mod * mod;

    return ratio * mod;
}

