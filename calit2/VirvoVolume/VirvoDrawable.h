#ifndef VIRVO_DRAWABLE
#define VIRVO_DRAWABLE

#define VV_LIBRARY_BUILD

#include <osg/Drawable>
#include <osg/Geode>
#include <map>
#include <virvo/vvvecmath.h>
#include <virvo/vvrenderer.h>

namespace osg
{
   class State;
};

class vvVolDesc;
class vvTransFunc;

class VirvoDrawable : public osg::Drawable
{
   public:
      VirvoDrawable();
      virtual ~VirvoDrawable();

      // drawable virtual functions
      VirvoDrawable(const VirvoDrawable &, const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY);
      virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
      virtual Object* cloneType() const { return NULL; }
      virtual Object* clone(const osg::CopyOp& copyop) const { return new VirvoDrawable(*this,copyop); }
      virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const VirvoDrawable*>(obj)!=NULL; }
      virtual const char* libraryName() const { return "Virvo"; }
      virtual const char* className() const { return "VirvoDrawable"; }

      // virvo functions
      void  setCurrentFrame(int);
      int   getCurrentFrame() const;
      int   getNumFrames() const;
      void  setViewDirection(const osg::Vec3&);
      void  setObjectDirection(const osg::Vec3&);

      void  setClipDirection(const osg::Vec3&);
      void  setClipPoint(const osg::Vec3&);
      osg::Vec3 getClipDirection() const;
      osg::Vec3 getClipPoint() const;
      void setClipping(bool enable);
      bool getClipping() const;
      void  setClipColor(const osg::Vec3&);
      osg::Vec3 getClipColor() const;
      void setSingleSliceClipping(bool enable);

      void  setROIPosition(const osg::Vec3&);
      void  setROISize(float);
      void  setROISelected(bool value);
      osg::Vec3 getROIPosition() const;
      float  getROISize() const;

      void setPosition(const osg::Vec3 &);
      osg::Vec3 getPosition() const;
      osg::Vec3 getCenter() const;
      void getBoundingBox(osg::Vec3 *min, osg::Vec3 *max) const;
      vvVolDesc *getVolumeDescription() const;
      void setVolumeDescription(vvVolDesc *vd);
      bool getInstantMode() const;
      void setPreintegration(bool value);
      void setInterpolation(bool value);
      void setBoundaries(bool value);
      void setQuality(float quality);
      float getQuality() const;
      void enableFlatDisplay( bool enable );

      void setTransferFunction(vvTransFunc *tf);

      enum BlendMode {
         AlphaBlend,
         MinimumIntensity,
         MaximumIntensity
      };

      void setBlendMode(BlendMode mode);
      BlendMode getBlendMode() const;

  protected:

      void init();
      void setParameter(const vvRenderState::ParameterType param, const vvParam& newValue);

      mutable vvVolDesc *vd;
      struct ContextState
      {
         ContextState();
         ~ContextState();
         vvRenderer *renderer;
         int transFuncCnt;  ///< draw process should update the transfer function if this is not equal to myUserData->transFuncCnt
         int lastDiscreteColors;
      };

      mutable std::map<int,ContextState> contextState;
      mutable int currentFrame;
      vvVector3 viewDir;
      vvVector3 objDir;
      bool preint;
      bool selected;
      bool interpolation;
      bool culled;
      vvRenderState renderState;
      bool flatDisplay;
      BlendMode blendMode;

};
#endif
