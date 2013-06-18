#ifndef _COPLANAR_LAYOUT_H_
#define _COPLANAR_LAYOUT_H_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>
#include <PluginMessageType.h>

#include <map>
#include <string>

class PlanarLayout : public cvr::CVRPlugin
{
public:        

    virtual ~PlanarLayout();

    bool init();

protected:

    class PlanarLayoutAlgorithm : public Layout
    {
    public:
        virtual std::string Name( void );
        virtual void Cleanup( void );
        virtual bool Start( void );
        virtual bool Update( void );

    protected:
        struct TransitionState
        {
            osg::Vec3 startPos;
            float startScale;
            osg::Vec3 endPos;
            float endScale;
        };

        typedef std::map< cvr::SceneObject*, TransitionState > SceneObjTrans;
        SceneObjTrans mObjTrans;

        double mTimeElapsed;

        bool GetBestGrid(unsigned int &r, unsigned int &c, unsigned int cells) const;
        double Rate(unsigned int r, unsigned int c, unsigned int cells) const;
    };

    PlanarLayoutAlgorithm* mLayoutAlgorithm;
};

#endif /*_COPLANAR_LAYOUT_H_*/

