#ifndef FP_GRAPH_LAYOUT_OBJECT_H
#define FP_GRAPH_LAYOUT_OBJECT_H

#include <cvrKernel/TiledWallSceneObject.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>

#include "GraphObject.h"

#include <osg/Geometry>
#include <osg/Geode>
#include <osgText/Text>

#include <ctime>
#include <vector>
#include <map>

class GraphLayoutObject : public cvr::TiledWallSceneObject
{
    public:
        GraphLayoutObject(float width, float height, int maxRows, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~GraphLayoutObject();

        void addGraphObject(GraphObject * object);
        void removeGraphObject(GraphObject * object);
        void removeAll();

        virtual void menuCallback(cvr::MenuItem * item);

    protected:
        void makeGeometry();
        void updateLayout();

        std::vector<GraphObject *> _objectList;
        std::map<GraphObject *,cvr::MenuButton *> _deleteButtonMap;

        float _width;
        float _height;
        int _maxRows;

        time_t _maxX;
        time_t _minX;
        time_t _currentMaxX;
        time_t _currentMinX;

        osg::ref_ptr<osg::Geode> _layoutGeode;

        cvr::MenuButton * _resetLayoutButton;
        cvr::MenuCheckbox * _syncTimeCB;
};

#endif
