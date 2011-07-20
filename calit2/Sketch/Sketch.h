#ifndef SKETCH_PLUGIN_H
#define SKETCH_PLUGIN_H

#include <kernel/CVRPlugin.h>

#include <menu/SubMenu.h>
#include <menu/MenuTextButtonSet.h>

#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Geometry>

#include <string>
#include <vector>
#include <map>

class Sketch : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        Sketch();
        virtual ~Sketch();

        bool init();
        void menuCallback(cvr::MenuItem * item);
        void preFrame();
        bool buttonEvent(int type, int button, int hand, const osg::Matrix & mat);
        bool mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix & mat);

    protected:
        enum DrawMode
        {
            NONE = -1,
            RIBBON,
            TUBE,
            SPHERE
        };

        struct MyComputeBounds : public osg::Drawable::ComputeBoundingBoxCallback
        {
            MyComputeBounds() {}
            MyComputeBounds(const MyComputeBounds & mcb, const osg::CopyOp &) {}
            virtual osg::BoundingBox computeBound(const osg::Drawable &) const;

            osg::BoundingBox _bound;
        };

        cvr::SubMenu * _sketchMenu;
        cvr::MenuTextButtonSet * _modeButtons;

        DrawMode _mode;

        std::string _dir;
        bool _drawing;

        float _brushScale;
        float _pointerDistance;

        osg::Matrix _lastTransform;

        osg::ref_ptr<osg::MatrixTransform> _brushRoot;
        std::vector<osg::ref_ptr<osg::Geode> > _brushes;

        osg::ref_ptr<osg::MatrixTransform> _sketchRoot;
        osg::ref_ptr<osg::Geode> _sketchGeode;

        osg::BoundingBox * _currentBound;

        osg::Vec4 _color;
        unsigned int _count;
        osg::Vec3Array * _verts;
        osg::Vec4Array * _colors;
        osg::Vec3Array * _normals;
        osg::DrawArrays * _primitive;
        osg::Geometry * _currentGeometry;

        std::map<double, osg::ref_ptr<osg::Geometry> > _geometryMap;
};

#endif
