/**
 * @file PreDrawHook.h
 * Contains classes that setup a predraw callback on the openscenegraph camera 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#ifndef PRE_DRAW_HOOK_H
#define PRE_DRAW_HOOK_H

#include <GL/glut.h>
#include <osg/Camera>

class PreDrawCallback;


/**
 * Attached to an osg camera.
 *
 * The operator() function is called every frame before the start of the scene draw, 
 * we use this to call our own callback function to abstract away the osg interface.
 */
class PreDrawHook : public osg::Camera::Camera::DrawCallback
{
    public:
        PreDrawHook(PreDrawCallback * callback);
        virtual ~PreDrawHook();

        virtual void operator()(osg::RenderInfo & ri) const;

    protected:
        PreDrawCallback * _callback;    ///< pointer to the class that gets a callback from this hook
};

/**
 * Interface class for our predraw callback.
 *
 * A class that implements this will get its preDrawCallback function called before every draw.
 */
class PreDrawCallback
{
    public:
        PreDrawCallback() {}
        virtual ~PreDrawCallback() {}

        /**
         * Function called before each draw, in our case, by the PreDrawHook on an osg camera
         *
         * @param cam osg camera used to collect stat information
         */
        virtual void preDrawCallback(osg::Camera * cam) = 0;
};

#endif
