#ifndef FP_GRAPH_LAYOUT_OBJECT_H
#define FP_GRAPH_LAYOUT_OBJECT_H

#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/MenuText.h>
#include <cvrMenu/PopupMenu.h>
#include <cvrInput/TrackerBase.h>

#include "GraphObject.h"
#include "MicrobeGraphObject.h"
#include "MicrobeBarGraphObject.h"
#include "GraphKeyObject.h"
#include "FPTiledWallSceneObject.h"

#include <osg/Geometry>
#include <osg/Geode>
#include <osgText/Text>

#include <ctime>
#include <vector>
#include <map>

class GraphLayoutObject : public FPTiledWallSceneObject
{
    public:
        GraphLayoutObject(float width, float height, int maxRows, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds=false);
        virtual ~GraphLayoutObject();

        void addGraphObject(LayoutTypeObject * object);
        void removeGraphObject(LayoutTypeObject * object);

        void addLineObject(LayoutLineObject * object);
        void removeLineObject(LayoutLineObject * object);

        void selectMicrobes(std::string & group, std::vector<std::string> & keys);
        void selectPatients(std::string & group, std::vector<std::string> & patients);

        void removeAll();
        void perFrame();

        void minimize();
        void maximize();
        bool isMinimized()
        {
            return _minimized;
        }

        void setRows(float rows);
        void setSyncTime(bool sync);

        GraphKeyObject * getPatientKeyObject()
        {
            return _patientKey;
        }

        GraphKeyObject * getPhylumKeyObject()
        {
            return _phylumKey;
        }

        bool dumpState(std::ostream & out);
        bool loadState(std::istream & in);

        virtual void menuCallback(cvr::MenuItem * item);

        virtual bool processEvent(cvr::InteractionEvent * event);
        virtual void enterCallback(int handID, const osg::Matrix &mat);
        virtual void updateCallback(int handID, const osg::Matrix &mat);
        virtual void leaveCallback(int handID);

        void setChartLinearRegression(bool lr);
        void setSingleMicrobeLogScale(bool logScale);
        void setSingleMicrobeShowStdDev(bool value);
        void setScatterLogScale(bool logScale);

        void forceUpdate();
    protected:
        void makeGeometry();
        void makeKeys();
        void updateGeometry();
        void updateLayout();
        void checkLineRefs();
        void setTitle(std::string title);
        void setSelectionText();

        bool loadObject(std::istream & in);

        bool _minimized;

        std::vector<LayoutTypeObject *> _objectList;
        std::map<LayoutTypeObject *,cvr::MenuButton *> _deleteButtonMap;
        std::vector<LayoutLineObject *> _lineObjectList;

        std::string _currentSelectedMicrobeGroup;
        std::vector<std::string> _currentSelectedMicrobes;
        std::string _currentSelectedPatientGroup;
        std::vector<std::string> _currentSelectedPatients;
        std::map<std::string,std::vector<std::string> > _currentSelectedPatientMap;

        GraphKeyObject * _phylumKey;
        GraphKeyObject * _patientKey;

        float _width;
        float _height;
        int _maxRows;

        time_t _maxX;
        time_t _minX;
        time_t _currentMaxX;
        time_t _currentMinX;

        osg::ref_ptr<osg::Geode> _layoutGeode;
        osg::ref_ptr<osg::Vec3Array> _verts;
        osg::ref_ptr<osgText::Text> _text;

        cvr::MenuButton * _resetLayoutButton;
        cvr::MenuButton * _minmaxButton;
        cvr::MenuCheckbox * _syncTimeCB;
        cvr::MenuCheckbox * _zoomCB;
        cvr::MenuRangeValueCompact * _rowsRV;
        cvr::MenuRangeValueCompact * _widthRV;
        cvr::MenuRangeValueCompact * _heightRV;
        cvr::MenuButton * _removeUnselected;
        cvr::MenuCheckbox * _multiSelect;
        cvr::MenuButton * _linRegSort;

        cvr::PopupMenu * _selectionMenu;
        cvr::MenuText * _selectionText;

        int _activeHand;
        cvr::TrackerBase::TrackerType _activeHandType;

        std::vector<int> _perGraphActiveHand;
        std::vector<cvr::TrackerBase::TrackerType> _perGraphActiveHandType;
};

#endif
