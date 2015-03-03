#ifndef FP_LAYOUT_INTERFACES_H
#define FP_LAYOUT_INTERFACES_H

#include <cvrMenu/MenuCheckbox.h>

#include "FPTiledWallSceneObject.h"

#include <map>

class LayoutTypeObject : public FPTiledWallSceneObject
{
    public:
        LayoutTypeObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false) : FPTiledWallSceneObject(name,navigation,movable,clip,contextMenu,showBounds)
        {
        }
        virtual ~LayoutTypeObject()
        {
        }

        virtual void objectAdded()
        {
        }

        virtual void objectRemoved()
        {
        }

        virtual void setGraphSize(float width, float height) = 0;
        virtual bool getLayoutDoesDelete()
        {
            return false;
        }
        virtual void perFrame()
        {
        }
        virtual void setGLScale(float scale)
        {
        }
        virtual void setBarVisible(bool vis)
        {
        }
        virtual float getBarPosition()
        {
            return 0.0;
        }
        virtual void setBarPosition(float pos)
        {
        }
        virtual  bool getGraphSpacePoint(const osg::Matrix & mat, osg::Vec3 & point)
        {
            return false;
        }
        virtual void dumpState(std::ostream & out)
        {
            out << "UNKNOWN" << std::endl;
        }
        virtual bool loadState(std::istream & in)
        {
            return true;
        }
        virtual std::string getTitle()
        {
            return "";
        }
};

class LayoutLineObject : public FPTiledWallSceneObject
{
    public:
        LayoutLineObject(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false) : FPTiledWallSceneObject(name,navigation,movable,clip,contextMenu,showBounds)
        {
        }
        virtual ~LayoutLineObject()
        {
        }

        virtual void setSize(float width, float height) = 0;

        void ref(LayoutTypeObject * object);
        void unref(LayoutTypeObject * object);
        void unrefAll();
        bool hasRef();

    protected:
        std::map<LayoutTypeObject*,bool> _refMap;
};

class TimeRangeObject
{
    public:
        virtual void setGraphDisplayRange(time_t start, time_t end) = 0;
        virtual void getGraphDisplayRange(time_t & start, time_t & end) = 0;
        virtual void resetGraphDisplayRange() = 0;
        virtual time_t getMaxTimestamp() = 0;
        virtual time_t getMinTimestamp() = 0;
};

class ValueRangeObject
{
    public:
        virtual float getGraphMaxValue() = 0;
        virtual float getGraphMinValue() = 0;

        virtual float getGraphDisplayRangeMax() = 0;
        virtual float getGraphDisplayRangeMin() = 0;

        virtual void setGraphDisplayRange(float min, float max) = 0;
        virtual void resetGraphDisplayRange() = 0;
};

class LogValueRangeObject
{
    public:
        virtual float getGraphXMaxValue() = 0;
        virtual float getGraphXMinValue() = 0;
        virtual float getGraphZMaxValue() = 0;
        virtual float getGraphZMinValue() = 0;

        virtual float getGraphXDisplayRangeMax() = 0;
        virtual float getGraphXDisplayRangeMin() = 0;
        virtual float getGraphZDisplayRangeMax() = 0;
        virtual float getGraphZDisplayRangeMin() = 0;

        virtual void setGraphXDisplayRange(float min, float max) = 0;
        virtual void setGraphZDisplayRange(float min, float max) = 0;
        virtual void resetGraphDisplayRange() = 0;
};

class MicrobeSelectObject
{
    public:
        virtual void selectMicrobes(std::string & group, std::vector<std::string> & keys) = 0;
        virtual float getGroupValue(std::string group, int i) = 0;
        virtual float getMicrobeValue(std::string group, std::string key, int i) = 0;
        virtual int getNumDisplayValues() = 0;
        virtual std::string getDisplayLabel(int i) = 0;
};

class PatientSelectObject
{
    public:
        //virtual void selectPatients(std::string & group, std::vector<std::string> & patients) = 0;
        virtual void selectPatients(std::map<std::string,std::vector<std::string> > & selectMap) = 0;
};

class SelectableObject
{
    public:
        SelectableObject()
        {
            _selectCB = NULL;
        }

        bool isSelected()
        {
            if(_selectCB && _selectCB->getValue())
            {
                return true;
            }
            return false;
        }

    protected:
        cvr::MenuCheckbox * _selectCB;
};
#endif
