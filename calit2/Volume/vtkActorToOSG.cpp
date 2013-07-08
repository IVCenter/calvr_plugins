//C++ source - fIVE|Analyse - Copyright (C) 2002-2003 Michael Gronager, UNI-C
//Distributed under the terms of the GNU Library General Public License (LGPL)
//as published by the Free Software Foundation.

// this is a workaround for compiling VTK with stlport
// stlport is required for compiling OSG on MSVS60 - (try .NET?)
#define _INC_STRSTREAM
#include <strstream>
// workaround end

#include "vtkActorToOSG.h"

#include <osg/Vec3>

osg::Geode* vtkActorToOSG(vtkActor *actor, osg::Geode *geode, int verbose) {

	// make actor current
	actor->GetMapper()->Update();

	// this could possibly be any type of DataSet, vtkActorToOSG assumes polyData
	if (strcmp(actor->GetMapper()->GetInput()->GetClassName(), "vtkPolyData")) {
		std::cerr << "ERROR! Actor must use a vtkPolyDataMapper." << std::endl;
		std::cerr << "If you are using a vtkDataSetMapper, use vtkGeometryFilter instead." << std::endl;
		return NULL;
	}

	// if geode doesn't exist, then create a new one
	if (geode == NULL)
		geode = new osg::Geode();

	// get poly data
	vtkPolyData *polyData = (vtkPolyData *) actor->GetMapper()->GetInput();

	// get primitive arrays
	osg::Geometry* points, *lines, *polys, *strips;

	// create new Geometry for the Geode
	points = processPrimitive(actor, polyData->GetVerts(), osg::PrimitiveSet::POINTS, verbose);
	lines = processPrimitive(actor, polyData->GetLines(), osg::PrimitiveSet::LINE_STRIP, verbose);
	polys = processPrimitive(actor, polyData->GetPolys(), osg::PrimitiveSet::POLYGON, verbose);
	strips = processPrimitive(actor, polyData->GetStrips(), osg::PrimitiveSet::TRIANGLE_STRIP, verbose);

	// remove old gsets and delete them
	while( geode->getNumDrawables() ) geode->removeDrawable((int)0);

	if( points ) geode->addDrawable( points );
	if( lines ) geode->addDrawable( lines );
	if( polys ) geode->addDrawable( polys );
	if( strips ) geode->addDrawable( strips );

	return geode;
}


osg::Geometry *processPrimitive(vtkActor *actor, vtkCellArray *primArray, int primType, int verbose) {

	// get polyData from vtkActor
	vtkPolyData *polyData = (vtkPolyData *) actor->GetMapper()->GetInput();

	int numPrimitives = primArray->GetNumberOfCells();
	if (numPrimitives == 0) 
		return NULL;

	//Initialize the Geometry
	osg::Geometry* geom = new osg::Geometry;

	// get number of indices in the vtk prim array. Each vtkCell has the length
	// (not counted), followed by the indices.
	int primArraySize = primArray->GetNumberOfConnectivityEntries();
	int numIndices = primArraySize - numPrimitives;

	// allocate as many verts as there are indices in vtk prim array
	osg::Vec3Array* vertices = new osg::Vec3Array;

	// check to see if there are normals
	int normalPerVertex = 0;
	int normalPerCell = 0;
	vtkDataArray* normals = NULL;
	
	normals = polyData->GetPointData()->GetNormals();
	//if (actor->GetProperty()->GetInterpolation() == VTK_FLAT)
	//	normals = NULL;
	if (normals != NULL)
    {
		normalPerVertex = 1;
    }
	else
	{
		normals = polyData->GetCellData()->GetNormals();
		if (normals != NULL)
			normalPerCell = 1;
	}
	

	osg::Vec3Array* norms = new osg::Vec3Array;

	// check to see if there is color information
	int colorPerVertex = 0;
	int colorPerCell = 0;
	vtkUnsignedCharArray *colorArray = actor->GetMapper()->MapScalars(1.0);
	if (actor->GetMapper()->GetScalarVisibility() && colorArray != NULL)
	{
		int scalarMode = actor->GetMapper()->GetScalarMode();
		if (scalarMode == VTK_SCALAR_MODE_USE_CELL_DATA ||
			!polyData->GetPointData()->GetScalars()) // there is no point data
			colorPerCell = 1;
		else
			colorPerVertex = 1;
	}

	osg::Vec4Array* colors = new osg::Vec4Array;

	// check to see if there are texture coordinates
	vtkDataArray *texCoords = polyData->GetPointData()->GetTCoords();
	osg::Vec2Array* tcoords = new osg::Vec2Array;

	// copy data from vtk prim array to osg Geometry
	int prim = 0, vert = 0;
	//int i, npts, totpts = 0, *pts;
	int i, totpts = 0;
	vtkIdType npts, *pts;

	// go through cells (primitives)
	for (primArray->InitTraversal(); primArray->GetNextCell(npts, pts); prim++)
	{ 
		geom->addPrimitiveSet(new osg::DrawArrays(primType, totpts, npts));
		totpts += npts;
		if (colorPerCell)
		{
			unsigned char *aColor = colorArray->GetPointer(4*prim);
	        colors->push_back(osg::Vec4(aColor[0]/255.0f, aColor[1]/255.0f,
										aColor[2]/255.0f, aColor[3]/255.0f));
		}
		if (normalPerCell)
		{
			double *aNormal = normals->GetTuple(prim);
			norms->push_back(osg::Vec3(aNormal[0], aNormal[1], aNormal[2]));
		}
		// go through points in cell (verts)
		for (i=0; i < npts; i++)
		{
			double *aVertex = polyData->GetPoint(pts[i]);
			vertices->push_back(osg::Vec3(aVertex[0], aVertex[1], aVertex[2]));
			if (normalPerVertex)
			{
				double *aNormal = normals->GetTuple(pts[i]);
				norms->push_back(osg::Vec3(aNormal[0], aNormal[1], aNormal[2]));
			}
			if (colorPerVertex)
			{  
				unsigned char *aColor = colorArray->GetPointer(4*pts[i]);
				//colors->push_back(osg::Vec4(aColor[0]/255.0f, aColor[1]/255.0f,
				//							aColor[2]/255.0f, aColor[3]/255.0f));
				colors->push_back(osg::Vec4(aColor[0]/255.0f, aColor[1]/255.0f,
											aColor[2]/255.0f, 0.4f));
			}
			if (texCoords != NULL)
			{
				double *aTCoord = texCoords->GetTuple(pts[i]);
				tcoords->push_back(osg::Vec2(aTCoord[0], aTCoord[1]));
			}
			vert++;
		}
	}

	// add attribute arrays to gset
	geom->setVertexArray(vertices);
    geom->setColorArray(colors);
    if(normals) geom->setNormalArray(norms);

	if (normalPerVertex)
        geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
	if (normalPerCell)
        geom->setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE);

	if (colorPerVertex)
        geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	else if (colorPerCell)
        geom->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);
	else
	{ 
    // use overall color (get from Actor)
		double *actorColor = actor->GetProperty()->GetColor();
		float opacity = actor->GetProperty()->GetOpacity();

		colors->push_back(osg::Vec4(actorColor[0], actorColor[1], actorColor[2], opacity));
        geom->setColorBinding(osg::Geometry::BIND_OVERALL);
	}
  
	if (texCoords != NULL)
		geom->setTexCoordArray(0,tcoords);


	// create a geostate for this geoset
	osg::StateSet* stateset = new osg::StateSet;

	// if not opaque
	if (actor->GetProperty()->GetOpacity() < 1.0)
	{
		stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN); // draw last
	    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
		stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
		stateset->setMode(GL_CULL_FACE,osg::StateAttribute::OFF);
	}
/*
	// wireframe - how do I set this ?
	if (actor->GetProperty()->GetRepresentation() == VTK_WIREFRAME) 
		stateset->setMode(GL_WIREFRAME, osg::StateAttribute::ON);
*/
	// backface culling
	if (!actor->GetProperty()->GetBackfaceCulling())
		stateset->setMode(GL_CULL_FACE,osg::StateAttribute::OFF);
  
	// lighting
	if (normals != NULL)
	    stateset->setMode(GL_LIGHTING,osg::StateAttribute::ON);
	else
	    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

	// if it is lines, turn off lighting. 
	if (primType == osg::PrimitiveSet::LINE_STRIP)
	    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    geom->setStateSet(stateset);
	return geom;
}

