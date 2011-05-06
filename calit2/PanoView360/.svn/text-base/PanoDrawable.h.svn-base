#ifndef _PANO_DRAWABLE_H_
#define _PANO_DRAWABLE_H_

#include <osg/Drawable>

class PanoDrawable : public osg::Drawable
{
    public:
        PanoDrawable() {}
        virtual ~PanoDrawable() {}

        virtual void drawImplementation(osg::RenderInfo&) const {}
        virtual void setFlip(int f) = 0;

        virtual void setImage(std::string file_path) = 0;
        virtual void setImage(std::string file_path_r, std::string file_path_l) = 0;

	virtual void updateRotate(float f) = 0;

        virtual float getRadius() = 0;
        virtual void setRadius(float r) = 0;
        virtual int getSegmentsPerTexture() = 0;
        virtual void setSegmentsPerTexture(int spt) = 0;
        virtual int getMaxTextureSize() = 0;
        virtual void setMaxTextureSize(int mts) = 0;
        virtual void getViewAngle(float & a, float & b) = 0;
        virtual void setViewAngle(float a, float b) = 0;
        virtual float getCamHeight() = 0;
        virtual void setCamHeight(float h) = 0;

        virtual void deleteTextures() = 0;
        virtual bool deleteDone() = 0;

        virtual Object* cloneType() const { return (Object *)0; }
        virtual Object* clone(const osg::CopyOp&) const { return (Object *)0; }
        virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const PanoDrawable*>(obj)!=NULL; }
        virtual const char* libraryName() const { return "Pano"; }
        virtual const char* className() const { return "PanoDrawable"; }

        virtual void setMap(std::map<std::string, std::map<int, std::vector<std::pair<std::pair<int, int>, int> > > > & map)
        {
            _eyeMap = map;
        }

        static int firsteye;

    protected:
        mutable std::map<std::string, std::map<int, std::vector<std::pair<std::pair<int, int>, int> > > > _eyeMap;

};

#endif
