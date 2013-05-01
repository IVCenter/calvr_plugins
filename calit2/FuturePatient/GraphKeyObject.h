#ifndef GRAPH_KEY_OBJECT_H
#define GRAPH_KEY_OBJECT_H

#include <osg/Vec4>
#include <osg/Geode>
#include <osg/Geometry>
#include <osgText/Text>

#include <vector>
#include <string>

#include "LayoutInterfaces.h"

class GraphKeyObject : public LayoutLineObject
{
    public:
        GraphKeyObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~GraphKeyObject();

        void setKeys(std::vector<osg::Vec4> & colors, std::vector<std::string> & labels);

        virtual void setSize(float width, float height);

    protected:
        void update();

        std::vector<osg::Vec4> _colors;
        std::vector<std::string> _labels;

        float _width, _height;

        osg::ref_ptr<osg::Geode> _geode;
        osg::ref_ptr<osg::Geometry> _bgGeom;
        osg::ref_ptr<osg::Geometry> _boxGeom;
        std::vector<osg::ref_ptr<osgText::Text> > _textList;

        osg::ref_ptr<osgText::Font> _font;
};

#endif
