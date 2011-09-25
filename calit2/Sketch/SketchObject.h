#ifndef SKETCH_OBJECT_H
#define SKETCH_OBJECT_H

#include <osg/Matrix>
#include <osg/Drawable>

#ifndef WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

class SketchObject
{
    public:
        SketchObject(osg::Vec4 color, float size) : _valid(false), _done(false),
             _size(size), _color(color) {}
        virtual ~SketchObject() {}

        virtual bool buttonEvent(int type, const osg::Matrix & mat) = 0;
        virtual void addBrush(osg::MatrixTransform * mt) = 0;
        virtual void removeBrush(osg::MatrixTransform * mt) = 0;
        virtual void updateBrush(osg::MatrixTransform * mt) = 0;
        virtual void finish() =  0;
        virtual osg::Drawable * getDrawable() = 0;
        virtual void setColor(osg::Vec4 color) { _color = color; }
        virtual void setSize(float size) { _size = size; }

        const osg::Vec4 & getColor() { return _color; }
        float getSize() { return _size; }
        void setDone(bool b) { _done = b; }
        bool isDone() { return _done; }
        void setValid(bool b) { _valid = b; }
        bool isValid() { return _valid; }
        void setTimeStamp(struct timeval & tv) { _timeStamp = tv; }
        const struct timeval & getTimeStamp() { return _timeStamp; }

        struct MyComputeBounds : public osg::Drawable::ComputeBoundingBoxCallback
        {
            MyComputeBounds() {}
            MyComputeBounds(const MyComputeBounds & mcb, const osg::CopyOp &) {}
            virtual osg::BoundingBox computeBound(const osg::Drawable &) const { return _bound; }

            osg::BoundingBox _bound;
        };

    protected:
        bool _done;
        bool _valid;
        osg::Vec4 _color;
        float _size;
        struct timeval _timeStamp;

};

#endif
