#ifndef PANOVIEW_LOD_H
#define PANOVIEW_LOD_H

#include <kernel/CVRPlugin.h>
#include "PanoDrawableLOD.h"

#include <osg/MatrixTransform>
#include <osg/Geode>

class PanoViewLOD : public cvr::CVRPlugin
{
    public:
	PanoViewLOD();
	virtual ~PanoViewLOD();

        bool init();

    protected:
        osg::MatrixTransform * _root;
        osg::Geode * _leftGeode;
        osg::Geode * _rightGeode;
        PanoDrawableLOD * _rightDrawable;
        PanoDrawableLOD * _leftDrawable;
};

#endif
