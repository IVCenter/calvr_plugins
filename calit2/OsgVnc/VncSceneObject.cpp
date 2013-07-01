#include "VncSceneObject.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/InteractionEvent.h>
#include <cvrUtil/OsgMath.h>

#include <osgText/Text>
#include <osg/Texture2D>
#include <osg/Geometry>


VncSceneObject::VncSceneObject(std::string name, osgWidget::VncClient* client, bool vncEvents, bool navigation, bool movable, bool clip,
                bool contextMenu, bool showBounds, float depth) : cvr::TiledWallSceneObject(name, navigation, movable, clip, contextMenu, showBounds), _client(client), _vncEvents(vncEvents), _depth(depth)
{
    // init activity
    _active = false;

	// create a intersectable frame widget (make it a 10th the height of the window)
    _height = 0.0;
    _width = 0.0;

    osg::StateAttribute* attr = _client->getDrawable(0)->getStateSet()->getTextureAttribute(0,osg::StateAttribute::TEXTURE);
    if (attr)
    {
        osg::Texture2D* texture2D = dynamic_cast<osg::Texture2D*>(attr);
        if (texture2D)
        {
            _image = dynamic_cast<osgWidget::VncImage* >(texture2D->getImage());
            _width = _image->s();
            _height = _image->t();
        }
    }

    _bound = _client->getDrawable(0)->getBound();

    // figure out scale
    _windowScale = (_bound.xMax() == 1.0) ?  _width: _height;

    osg::ref_ptr<osg::Geode> title = new osg::Geode();

    osg::Vec3Array * vertices = new osg::Vec3Array(4);
    (*vertices)[0].set(osg::Vec3(0.0f, _depth, 0.0f + _bound.zMax() + (_bound.zMax() * 0.1)));
    (*vertices)[1].set(osg::Vec3(0.0f, _depth, 0.0f + _bound.zMax()));
    (*vertices)[2].set(osg::Vec3(0.0f + _bound.xMax(), _depth, 0.0f + _bound.zMax()));
    (*vertices)[3].set(osg::Vec3(0.0f + _bound.xMax(), _depth, 0.0f + _bound.zMax() + (_bound.zMax() * 0.1)));

    osg::Vec4Array* colours = new osg::Vec4Array(1);
    (*colours)[0].set(135.0/255.0, 206.0/255.0, 250.0/255.0, 1.0f);

    osg::Geometry* geom = new osg::Geometry();
    geom->setVertexArray(vertices);
    geom->setColorArray(colours);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4));
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    title->addDrawable(geom);

    // create text title
    osgText::Text* text = new osgText::Text();
    text->setColor(osg::Vec4(1.0, 1.0, 1.0, 1.0));
    text->setCharacterSize(0.03f);
    text->setFont ("/usr/share/fonts/liberation/LiberationSans-Regular.ttf");
    text->setAxisAlignment(osgText::TextBase::XZ_PLANE);
    text->setAlignment(osgText::Text::CENTER_CENTER);
    text->setBackdropType(osgText::Text::NONE);
    text->setText(name);
    text->setPosition(osg::Vec3(_bound.xMax() * 0.5, _depth - 0.0001, _bound.zMax() + (_bound.zMax() * 0.05)));
    title->addDrawable(text);

	// attach the geodes to the custom scene object
	cvr::SceneObject::addChild(title);
    osg::ref_ptr<osg::MatrixTransform> offset = new osg::MatrixTransform();
    osg::Matrix mat;
    mat.setTrans(osg::Vec3(0, _depth, 0));
    offset->setMatrix(mat);
    offset->addChild(_client);
    cvr::SceneObject::addChild(offset);
}

bool VncSceneObject::processEvent(cvr::InteractionEvent * ie)
{
    // see if active and if should forward the event to the vnc server
    // add if only a gyromouse
	//if( _active && (ie->asPointerEvent()->getEventType() == cvr::POINTER_INTER_EVENT))
	if( _active )
	{
        if( _vncEvents )
        {
		    //check for button event now
		    cvr::TrackedButtonInteractionEvent* tie = ie->asTrackedButtonEvent();
            
            //check for pointerEvent
            //cvr::PointerInteractionEvent* tie = ie->asPointerEvent();
   		    if( tie )
   		    {
                // init button mask
                int buttonMask = 0;

                // convert cvr button to rfb button //TODO clean up
                if( tie->getButton() == 0  )
                    buttonMask |= 1;
            
                if( tie->getButton() == 1  )
                    buttonMask |= 4;
            
                if( tie->getButton() == 2  )
                    buttonMask |= 2;

                //check for double click (ignore double right click) //TODO add mouse wheel
                if ( tie->getInteraction() == cvr::BUTTON_DOUBLE_CLICK )
                {
                    _image->sendPointerEvent(_intersect.x(), _intersect.z(), buttonMask);
                    _image->sendPointerEvent(_intersect.x(), _intersect.z(), 0);
                    _image->sendPointerEvent(_intersect.x(), _intersect.z(), buttonMask);
                    _image->sendPointerEvent(_intersect.x(), _intersect.z(), 0);
                }
                else if ( tie->getInteraction() == cvr::BUTTON_UP )
                {
                    _image->sendPointerEvent(_intersect.x(), _intersect.z(), 0);
                }
                else 
                {
                    _image->sendPointerEvent(_intersect.x(), _intersect.z(), buttonMask);
                }

		    }
            else // mouse movement
            {
                _image->sendPointerEvent(_intersect.x(), _intersect.z(), 0);
            }
        }
		return true;
	}

    // process the event normally
   	return TiledWallSceneObject::processEvent(ie);
}

void VncSceneObject::updateCallback(int handID, const osg::Matrix & mat)
{
        // reset activity
        _active = false;

        // compute intersection point create event and send it to the vnc widget
		osg::Vec3 pointerStart, pointerEnd;
    	pointerStart = mat.getTrans() * getWorldToObjectMatrix();
    	pointerEnd.set(0.0f, 10000.0f, 0.0f);
    	pointerEnd = pointerEnd * mat * getWorldToObjectMatrix();

        osg::Vec3 point(0, 0, 0);
        osg::Vec3 planeNormal(0,-1,0);
        osg::Vec3 intersect;
        float w;

        if(cvr::linePlaneIntersectionRef(pointerStart,pointerEnd,point,planeNormal,intersect,w))
        {
            // make sure instersection is within bounds of image
            if( (intersect.x() >= 0.0) && (intersect.x() <= _bound.xMax()) &&
                (intersect.z() >= 0.0) && (intersect.z() <= _bound.zMax()) )
            {
                // translate into pixel co-ordinates
                _intersect.x() = intersect.x() * _windowScale;
                _intersect.z() = _height - (intersect.z() * _windowScale);

                _active = true;
            }
        }
}

VncSceneObject::~VncSceneObject()
{
    _active = false;

    // close client connection
    if( _client )
    {
        _client->close();
    }
}
