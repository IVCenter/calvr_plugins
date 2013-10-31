#include "VirvoDrawable.h"

#include <virvo/vvrendererfactory.h>
#include <virvo/vvrenderer.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvfileio.h>
#include <virvo/vvtexrend.h>

#include <iostream>

using namespace osg;
using namespace std;

#undef VERBOSE

VirvoDrawable::ContextState::ContextState()
{
   renderer=NULL;
   transFuncCnt=0;
   lastDiscreteColors=0;
}


VirvoDrawable::ContextState::~ContextState()
{
   //delete renderer;
}


VirvoDrawable::VirvoDrawable()
{
#ifdef VERBOSE
   cerr << "VirvoDrawable::<init> warn: empty constructor called" << endl;
#endif
   init();
}


VirvoDrawable::VirvoDrawable(const VirvoDrawable & drawable,
const osg::CopyOp & copyop)
: Drawable(drawable, copyop)
{
#ifdef VERBOSE
   cerr << "VirvoDrawable::<init> copying" << endl;
#endif
   init();
}


void VirvoDrawable::init()
{
   vd = NULL;
   setSupportsDisplayList(false);
   preint = false;
   interpolation = true;
   selected = false;
   flatDisplay = false;
   blendMode = AlphaBlend;
   currentFrame = 0;

   // disable clip perimeter
   setParameter(vvRenderState::VV_CLIP_PERIMETER, false);
}

// todo make sure this update occurs in the updatecallback
void VirvoDrawable::setParameter(const vvRenderState::ParameterType param, const vvParam& newValue)
{
    renderState.setParameter(param, newValue);
    std::map<int, ContextState>::iterator it;
    for( it = contextState.begin(); it != contextState.end(); ++it )
    {
        vvRenderer *&renderer = it->second.renderer;
        if(renderer)
        {
            renderer->setParameter(param, newValue);
        }
    }
}

VirvoDrawable::~VirvoDrawable()
{
#ifdef VERBOSE
   cerr << "VirvoDrawable::<dtor>: this=" << this << endl;
#endif
   contextState.clear();

   delete vd;
}

void VirvoDrawable::drawImplementation(RenderInfo &renderInfo) const
{
   int context = renderInfo.getContextID();

   // initalize it 
   if( contextState.find(context) == contextState.end() )
   {
       ContextState state;
       state.renderer = NULL;
       contextState[context] = state;
   }

   vvRenderer *&renderer = contextState[context].renderer;
   
   if(vd && !renderer)
   {
      // create renderer
      renderer = new vvTexRend(vd, renderState, vvTexRend::VV_VIEWPORT, vvTexRend::VV_FRG_PRG);
      if(renderer)
      {
          renderer->setParameter(vvRenderState::VV_BOUNDARIES, false);
          renderer->setParameter(vvRenderState::VV_ROI_POS,  vvVector3(0., 0., 0.));
          renderer->setParameter(vvRenderState::VV_ROI_SIZE, vvVector3(0., 0., 0.));
          renderer->setParameter(vvRenderState::VV_IS_ROI_USED, false);
          renderer->setParameter(vvRenderState::VV_MIP_MODE, 0);
          renderer->setParameter(vvRenderState::VV_OPAQUE_GEOMETRY_PRESENT, true);
          
          renderer->setParameter(vvRenderer::VV_SLICEINT, interpolation);
          renderer->setParameter(vvRenderer::VV_PREINT, preint);
          renderer->setParameter( vvTexRend::VV_SLICEORIENT, (int)flatDisplay );
       }
   }

   // if a renderer exists, process regular rendering procedure
   if (renderer)
   {
      ref_ptr<StateSet> currentState = new StateSet;
      renderInfo.getState()->captureCurrentState(*currentState);
      renderInfo.getState()->pushStateSet(currentState.get());

      renderer->setCurrentFrame(currentFrame);
      //renderer->setQuality(quality);
      renderer->setViewingDirection(viewDir);
      renderer->setObjectDirection(objDir);
      
      renderer->renderVolumeGL();
      renderInfo.getState()->popStateSet();
   }
   else
   {
      cerr << "vd==NULL" << endl;
   }
}


/// Caller: draw process

void  VirvoDrawable::setCurrentFrame(int frame)
{
   currentFrame = frame;
}

int VirvoDrawable::getCurrentFrame() const
{
   return currentFrame;
}

int VirvoDrawable::getNumFrames() const
{
   if(vd)
   {
      return vd->frames;
   }
   return 0;
}

void  VirvoDrawable::setViewDirection(const osg::Vec3& dir)
{
   viewDir = vvVector3(dir[0], dir[1], dir[2]);
}

void  VirvoDrawable::setObjectDirection(const osg::Vec3& dir)
{
   objDir = vvVector3(dir[0], dir[1], dir[2]);
}

void  VirvoDrawable::setClipDirection(const osg::Vec3& dir)
{ 
   setParameter(vvRenderState::VV_CLIP_PLANE_NORMAL,  vvVector3(dir[0], dir[1], dir[2]));
}

void  VirvoDrawable::setClipPoint(const osg::Vec3& point)
{
   setParameter(vvRenderState::VV_CLIP_PLANE_POINT,  vvVector3(point[0], point[1], point[2]));
}

void  VirvoDrawable::setClipColor(const osg::Vec3& color)
{
   setParameter(vvRenderState::VV_CLIP_COLOR,  vvColor(color[0], color[1], color[2]));
}

void VirvoDrawable::setClipping(bool enable)
{
   unsigned int mode = (enable) ? 1 : 0;
   setParameter(vvRenderState::VV_CLIP_MODE, mode);
}

osg::Vec3 VirvoDrawable::getClipDirection() const
{
    vvVector3 vec = renderState.getParameter(vvRenderState::VV_CLIP_PLANE_NORMAL);
    return osg::Vec3(vec[0], vec[1], vec[2]);
}

osg::Vec3 VirvoDrawable::getClipPoint() const
{
    vvVector3 vec = renderState.getParameter(vvRenderState::VV_CLIP_PLANE_POINT);    
    return osg::Vec3(vec[0], vec[1], vec[2]);
}

osg::Vec3 VirvoDrawable::getClipColor() const
{
    vvColor vec = renderState.getParameter(vvRenderState::VV_CLIP_COLOR);    
    return osg::Vec3(vec[0], vec[1], vec[2]);
}


bool VirvoDrawable::getClipping() const
{
   return renderState.getParameter(vvRenderState::VV_CLIP_MODE);
}

void VirvoDrawable::setSingleSliceClipping(bool enable)
{
   setParameter(vvRenderState::VV_CLIP_SINGLE_SLICE, enable);
}

void  VirvoDrawable::setROIPosition(const osg::Vec3& pos)
{
#ifdef VERBOSE
      cerr << "ROI pos: " << pos[0] << endl;
#endif
   setParameter(vvRenderState::VV_ROI_POS, vvVector3(pos[0], pos[1], pos[2]));
}

void  VirvoDrawable::setROISize(float size)
{
#ifdef VERBOSE
   cerr << "ROI size: " << size << endl;
#endif
   
   setParameter(vvRenderState::VV_ROI_SIZE, vvVector3(size, size, size));
   setParameter(vvRenderState::VV_IS_ROI_USED, (size > 0.f));
}

float VirvoDrawable::getROISize() const
{
   return renderState.getParameter(vvRenderState::VV_ROI_SIZE).asVec3()[0];
}

osg::Vec3 VirvoDrawable::getROIPosition() const
{
   vvVector3 roiPos = renderState.getParameter(vvRenderState::VV_ROI_POS);
   return osg::Vec3(roiPos[0], roiPos[1], roiPos[2]);
}

osg::Vec3 VirvoDrawable::getPosition() const
{
   if(vd)
   {
      return Vec3(vd->pos[0], vd->pos[1], vd->pos[2]);
   }
   else
      return Vec3(0., 0., 0.);
}

void VirvoDrawable::setPosition(const osg::Vec3 &pos)
{
   if(vd)
   {
      vd->pos = vvVector3(pos[0], pos[1], pos[2]);
   }
}

void VirvoDrawable::getBoundingBox(osg::Vec3 *min, osg::Vec3 *max) const
{
  const BoundingBox &bb = getBound();
  *min = bb._min;
  *max = bb._max;
}

vvVolDesc *VirvoDrawable::getVolumeDescription() const
{
   return vd;
}

void VirvoDrawable::setVolumeDescription(vvVolDesc *v)
{
	//printf("before this false \n");
   contextState.clear();
   delete vd;

#ifdef VERBOSE
   fprintf(stderr,  "setVolumeDescription: voldesc = %p\n", v );
#endif

   vd = v;

   if (vd && vd->tf.isEmpty())
   {
      vd->tf.setDefaultColors(vd->chan==1 ? 0 : 3, 0., 1.);
      vd->tf.setDefaultAlpha(0, 0., 1.);
   }

   dirtyBound();

   if(vd)
   {
      osg::Vec3 diag = osg::Vec3(vd->vox[0]*vd->dist[0], vd->vox[1]*vd->dist[1], vd->vox[2]*vd->dist[2]);
      setInitialBound(BoundingBox(getPosition()-diag*.5, getPosition()+diag*.5));
   }
   else
   {
      setInitialBound(BoundingBox(Vec3(0.,0.,0.), Vec3(0.,0.,0.)));
   }
}

bool VirvoDrawable::getInstantMode() const
{
   if(contextState.size()>0)
   {
      vvRenderer *&renderer = contextState[0].renderer;
      if(renderer)
      {
         return renderer->instantClassification();
      }
      else
      {
#ifdef VERBOSE
         fprintf(stderr, "instant false\n");
#endif
         return false;
      }
         
   }
   else
   {
      return true;
   }
}

void VirvoDrawable::setPreintegration(bool val)
{
   preint = val;
   setParameter(vvRenderer::VV_PREINT, preint);
}

void VirvoDrawable::setROISelected(bool val)
{
   selected = val;
   if (selected)
   {
        setParameter(vvRenderState::VV_PROBE_COLOR, vvVector3(1.0f, 0.0f, 0.0f));
   }
   else
   {
		setParameter(vvRenderState::VV_PROBE_COLOR, vvVector3(1.0f, 1.0f, 1.0f));
   }
}

void VirvoDrawable::setInterpolation(bool val)
{
   interpolation = val;
   setParameter(vvRenderer::VV_SLICEINT, interpolation);
}

void VirvoDrawable::setBoundaries(bool val)
{
   setParameter(vvRenderState::VV_BOUNDARIES, val);
}

void VirvoDrawable::setQuality(float q)
{
   setParameter(vvRenderState::VV_QUALITY, q);
}

float VirvoDrawable::getQuality() const
{
   return renderState.getParameter(vvRenderState::VV_QUALITY);
}

void VirvoDrawable::setTransferFunction(vvTransFunc *tf)
{
   if(vd && tf)
   {
      vvTransFunc::copy(&vd->tf._widgets, &tf->_widgets);
      vd->tf.setDiscreteColors(tf->getDiscreteColors());
   }

   for(unsigned int i=0; i<contextState.size(); i++)
   {
      vvRenderer *&renderer = contextState[i].renderer;
      if(renderer)
         renderer->updateTransferFunction();
   }
}

void VirvoDrawable::enableFlatDisplay(bool enable)
{
   flatDisplay = enable;
   setParameter( vvTexRend::VV_SLICEORIENT, (float)flatDisplay );
}

void VirvoDrawable::setBlendMode(BlendMode mode)
{
   blendMode = mode;
   if(blendMode == AlphaBlend)
   {
      setParameter(vvRenderState::VV_MIP_MODE, 0);
   }
   else if(blendMode == MaximumIntensity)
   {
      setParameter(vvRenderState::VV_MIP_MODE, 1);
   }
   else if(blendMode == MinimumIntensity)
   {
      setParameter(vvRenderState::VV_MIP_MODE, 2);
   }
}

VirvoDrawable::BlendMode VirvoDrawable::getBlendMode() const
{
   return blendMode;
}
