#ifndef _PANO_DRAWABLE_H_
#define _PANO_DRAWABLE_H_

#include <osg/Drawable>

class PanoDrawable : public osg::Drawable
{
    public:
        PanoDrawable(float radius_in, float viewanglev_in, float viewangleh_in, float camHeight_in, int segmentsPerTexture_in, int maxTextureSize_in);
        PanoDrawable(const PanoDrawable&,const osg::CopyOp&);
        virtual ~PanoDrawable();

        virtual void drawImplementation(osg::RenderInfo&) const;
        virtual void setFlip(int f);

        virtual void setImage(std::string file_path);
        virtual void setImage(std::string file_path_r, std::string file_path_l);

        virtual void updateRotate(float f);

        virtual float getRadius();
        virtual void setRadius(float r);
        virtual int getSegmentsPerTexture();
        virtual void setSegmentsPerTexture(int spt);
        virtual int getMaxTextureSize();
        virtual void setMaxTextureSize(int mts);
        virtual void getViewAngle(float & a, float & b);
        virtual void setViewAngle(float a, float b);
        virtual float getCamHeight();
        virtual void setCamHeight(float h);

        virtual void deleteTextures();
        virtual bool deleteDone();

        virtual Object* cloneType() const { return (Object *)0; }
        virtual Object* clone(const osg::CopyOp&) const { return (Object *)0; }
        virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const PanoDrawable*>(obj)!=NULL; }
        virtual const char* libraryName() const { return "Pano"; }
        virtual const char* className() const { return "PanoDrawable"; }

        enum eye
        {
            RIGHT = 1,
            LEFT = 2,
            BOTH = 3
        };

        virtual void drawShape(PanoDrawable::eye eye, int context) const;
    protected:

        mutable int currenteye;

        mutable int badinit;

        bool initTexture(eye e, int context) const;

        float _rotation;

        static OpenThreads::Mutex _initLock;
        static OpenThreads::Mutex _leftLoadLock;
        static OpenThreads::Mutex _rightLoadLock;
        static OpenThreads::Mutex _singleLoadLock;
        static OpenThreads::Mutex _rcLock;

        bool _useSingleLock;
        bool _highRamLoad;

        mutable bool _doDelete;
        static bool _deleteDone;
        mutable int rows, cols; 
        float radius;
        float viewanglev, viewangleh;
        float camHeight, floorOffset;
        mutable std::string rfile, lfile;
        //GLuint * textures;
        mutable int segmentsPerTexture, maxTextureSize, width, height, mono, flip;

        mutable std::vector<std::vector< unsigned char * > > rtiles;
        mutable std::vector<std::vector< unsigned char * > > ltiles;
        static std::map<int, std::vector<std::vector< GLuint * > > > rtextures;
        static std::map<int, std::vector<std::vector< GLuint * > > > ltextures;
        static std::map<int, int> _contextinit;
        mutable int _maxContext;
        mutable int _eyeMask;

        bool _renderOnMaster;
};

#endif
