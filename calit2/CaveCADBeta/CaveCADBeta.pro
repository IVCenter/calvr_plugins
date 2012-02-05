!include($$(COFRAMEWORKDIR)/mkspecs/config-first.pri):error(include of config-first.pri failed)
### don't modify anything before this line ###

TARGET		= CaveCADBeta
PROJECT         = General
TEMPLATE        = opencoverplugin


DEFINES *= CAVECADBeta

CONFIG          *= coappl colib openvrui math vrml97

macx:!tiger:QMAKE_CXXFLAGS *= -fno-coalesce

SOURCES  =  CaveCADBeta.cpp
SOURCES +=  CAVEDesigner.cpp
SOURCES +=  DesignStateHandler.cpp
SOURCES +=  DesignStateRenderer.cpp
SOURCES +=  AudioConfigHandler.cpp
SOURCES +=  Audio/OSCPack.cpp
SOURCES +=  DesignStates/DesignStateParticleSystem.cpp
SOURCES +=  DesignStates/DesignStateBase.cpp
SOURCES +=  DesignStates/DSVirtualSphere.cpp
SOURCES +=  DesignStates/DSVirtualEarth.cpp
SOURCES +=  DesignStates/DSParamountSwitch.cpp
SOURCES +=  DesignStates/DSSketchBook.cpp
SOURCES +=  DesignStates/DSGeometryCreator.cpp
SOURCES +=  DesignStates/DSGeometryEditor.cpp
SOURCES +=  DesignStates/DSTexturePallette.cpp
SOURCES +=  DesignStateIntersector.cpp
SOURCES +=  DesignObjectHandler.cpp
SOURCES +=  DesignObjectIntersector.cpp
SOURCES +=  DesignObjects/DesignObjectBase.cpp
SOURCES +=  DesignObjects/DOGeometryCollector.cpp
SOURCES +=  DesignObjects/DOGeometryCreator.cpp
SOURCES +=  DesignObjects/DOGeometryEditor.cpp
SOURCES +=  DesignObjects/DOGeometryCloner.cpp
SOURCES +=  CAVEIntersector.cpp
SOURCES +=  TrackballController.cpp
SOURCES +=  SnapLevelController.cpp
SOURCES +=  VirtualScenicHandler.cpp
SOURCES +=  Geometry/CAVEGeometry.cpp	
SOURCES +=  Geometry/CAVEGroupReference.cpp
SOURCES +=  Geometry/CAVEGroupShape.cpp
SOURCES +=  Geometry/CAVEGroupIconSurface.cpp
SOURCES +=  Geometry/CAVEGroupIconToolkit.cpp
SOURCES +=  Geometry/CAVEGroupEditWireframe.cpp
SOURCES +=  Geometry/CAVEGroupEditGeodeWireframe.cpp
SOURCES +=  Geometry/CAVEGroupEditGeometryWireframe.cpp
SOURCES +=  Geometry/CAVEGeode.cpp
SOURCES +=  Geometry/CAVEGeodeShape.cpp
SOURCES +=  Geometry/CAVEGeodeIcon.cpp
SOURCES +=  Geometry/CAVEGeodeIconSurface.cpp
SOURCES +=  Geometry/CAVEGeodeIconToolkit.cpp
SOURCES +=  Geometry/CAVEGeodeReference.cpp
SOURCES +=  Geometry/CAVEGeodeSnapWireframe.cpp
SOURCES +=  Geometry/CAVEGeodeSnapSolidshape.cpp
SOURCES +=  Geometry/CAVEGeodeEditWireframe.cpp
SOURCES +=  AnimationModeler/AnimationModelerBase.cpp
SOURCES +=  AnimationModeler/ANIMDSParticleSystem.cpp
SOURCES +=  AnimationModeler/ANIMVirtualSphere.cpp
SOURCES +=  AnimationModeler/ANIMVirtualEarth.cpp
SOURCES +=  AnimationModeler/ANIMRefXYPlane.cpp
SOURCES +=  AnimationModeler/ANIMRefSkyDome.cpp
SOURCES +=  AnimationModeler/ANIMRefWaterSurf.cpp
SOURCES +=  AnimationModeler/ANIMParamountPaintFrames.cpp
SOURCES +=  AnimationModeler/ANIMSketchBook.cpp
SOURCES +=  AnimationModeler/ANIMGeometryCollector.cpp
SOURCES +=  AnimationModeler/ANIMGeometryCreator.cpp
SOURCES +=  AnimationModeler/ANIMGeometryEditor.cpp
SOURCES +=  AnimationModeler/ANIMTexturePallette.cpp
SOURCES +=  AnimationModeler/ANIMObjectHandler.cpp

EXTRASOURCES  = *.h

### don't modify anything below this line ###
!include ($$(COFRAMEWORKDIR)/mkspecs/config-last.pri):error(include of config-last.pri failed)
