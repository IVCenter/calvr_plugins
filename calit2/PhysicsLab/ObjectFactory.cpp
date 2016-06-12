#include "ObjectFactory.h"

#include <iostream>
#include <string>
#include <cmath>

#include <osgDB/WriteFile>

int customId = -1;

ObjectFactory::ObjectFactory() {
  bh = new BulletHandler();
  numObjects = 0;
  numLights = 1;
  grabbedPhysId = grabbedId = -1;
  grabbedMatrix = 0;
  m_wonGame = false;
}

ObjectFactory::~ObjectFactory() {
  delete bh;
}

MatrixTransform* ObjectFactory::addBox( Vec3 pos, Vec3 halfLengths, Quat quat, Vec4 diffuse, bool phys = true, bool render = true, bool grabable = true ) {
  MatrixTransform* mt = new MatrixTransform;
  
  if ( render ) {
    Geode * box = new Geode;
    Box * boxprim = new Box( Vec3(0,0,0), 1);
    boxprim->setHalfLengths( halfLengths );
    ShapeDrawable * sd = new ShapeDrawable(boxprim);
    sd->setColor( diffuse );
    box->addDrawable(sd);
    Matrix boxm;
    boxm.makeTranslate(pos);
    boxm.setRotate(quat);
    mt->setMatrix( boxm );
    mt->addChild( box );
    
    if (!grabable) box->setNodeMask(~2);
  }
  
  numObjects++;
  m_objects.push_back( mt );
  m_scales.push_back(1.0f);
  m_physid.push_back( bh->addBox( pos, halfLengths, quat, phys ) );
  
  return mt;
}

MatrixTransform* ObjectFactory::addSeesaw( Vec3 pos, Vec3 halfLengths, Vec4 diffuse, bool phys, bool render ) {
  MatrixTransform* mt = new MatrixTransform;
  
  if (render) {
    Geode * seesaw = new Geode;
    Box * seesawPrim = new Box( Vec3(0,0,0), 1,1,1 );
    seesawPrim->setHalfLengths( halfLengths );
    ShapeDrawable * seesawsd = new ShapeDrawable(seesawPrim);
    seesaw->addDrawable(seesawsd);
    seesawsd->setColor(diffuse);
    Matrixd ssm;
    ssm.makeTranslate( pos );
    mt->setMatrix( ssm );
    mt->addChild( seesaw );
  }
  
  numObjects++;
  m_objects.push_back( mt );
  m_scales.push_back(1.0f);
  m_physid.push_back( bh->addSeesaw( pos, halfLengths, phys ) );
  
  return mt;
}

void ObjectFactory::addInvisibleWall( Vec3 pos, Vec3 halfLengths, int collisionFlag ) {
  bh->addInvisibleWall( pos, halfLengths, collisionFlag );
}

MatrixTransform* ObjectFactory::addSphere( Vec3 pos, double radius, Vec4 diffuse, bool phys, bool render ) {
  MatrixTransform* mt = new MatrixTransform;
  
  if (render) {
    Matrixd spherem;
    spherem.makeTranslate(pos);
    mt->setMatrix( spherem );
    
    Geode * tsphere = new Geode;
    Sphere * tsphereprim = new Sphere( Vec3(0,0,0), radius);
    ShapeDrawable * sphered = new ShapeDrawable(tsphereprim);
    tsphere->getOrCreateStateSet()->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
    sphered->setColor( diffuse );
    tsphere->addDrawable(sphered);
    tsphere->setNodeMask(~2);
    mt->addChild(tsphere);
  }
  
  numObjects++;
  m_objects.push_back( mt );
  m_scales.push_back(1.0f);
  m_solvers.push_back( mt );
  m_physid.push_back( bh->addSphere( pos, radius, phys ) );
  
  return mt;
}

MatrixTransform* ObjectFactory::addCylinder( Vec3 pos, double radius, double height, Vec4 color, bool phys, bool render) {
  MatrixTransform* mt = new MatrixTransform;
  
  if (render) {
    Matrixd cylm;
    cylm.makeTranslate(pos);
    mt->setMatrix( cylm );
    
    Geode * tcyl = new Geode;
    Cylinder * tcylprim = new Cylinder( Vec3(0,0,0), radius, height);
    ShapeDrawable * cyld = new ShapeDrawable(tcylprim);
    tcyl->getOrCreateStateSet()->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
    cyld->setColor( color );
    tcyl->addDrawable(cyld);
    mt->addChild(tcyl);
  }
  
  numObjects++;
  m_objects.push_back( mt );
  m_scales.push_back(1.0f);
  m_physid.push_back( bh->addCylinder( pos, Vec3(height, radius, 0), phys ) );
  
  return mt;
}

MatrixTransform* ObjectFactory::addOpenBox( Vec3 pos, Vec3 halfLengths, double innerWidth, bool phys, bool render ) {
  MatrixTransform* mt = new MatrixTransform;
  
  if (render) {
    CompositeShape * cs = new CompositeShape();
    cs->addChild( new Box(- Vec3(halfLengths.x() - innerWidth / 2, 0, 0), innerWidth, halfLengths.y() * 2, halfLengths.z() * 2) );
    cs->addChild( new Box(Vec3(halfLengths.x() - innerWidth / 2, 0, 0), innerWidth, halfLengths.y() * 2, halfLengths.z() * 2) );
    cs->addChild( new Box(- Vec3(0, halfLengths.y() - innerWidth / 2, 0), halfLengths.x() * 2, innerWidth, halfLengths.z() * 2) );
    cs->addChild( new Box(Vec3(0, halfLengths.y() - innerWidth / 2, 0), halfLengths.x() * 2, innerWidth, halfLengths.z() * 2) );
    cs->addChild( new Box(- Vec3(0, 0, halfLengths.z() - innerWidth / 2), halfLengths.x() * 2, halfLengths.y() * 2, innerWidth) );
    
    Geode * obg = new Geode;
    ShapeDrawable* sd = new ShapeDrawable(cs);
    sd->setColor( Vec4f(1,1,1,1) );
    obg->setNodeMask(~2);
    obg->addDrawable( sd );
    mt->addChild( obg );
    Matrixd obm;
    obm.makeTranslate(pos);
    mt->setMatrix( obm );
    //osgDB::writeNodeFile( *obg, "open_pit.wrl");
    //exit(1);
  }
  
  numObjects++;
  m_objects.push_back( mt );
  m_scales.push_back(1.0f);
  m_physid.push_back( bh->addOpenBox( pos, halfLengths, innerWidth, phys ) );
  
  return mt;
}

MatrixTransform* ObjectFactory::addCustomObject( std::string path, double scale, Vec3 pos, Quat rot, bool phys, bool pickable, float restitution) {
  MatrixTransform* mt = new MatrixTransform;

  // cached models cause issue when the are different sizes
  Node *model = modelBank[path];
  if (!model) {
    model = osgDB::readNodeFile( path );
    modelBank[path] = model;
  } else {
    std::cout << "Model found in cache." << std::endl;
  }
  
  Matrixd m;
  m.setTrans(pos);
  m.setRotate(rot);
  mt->setMatrix(m);
  
  if (model != NULL) 
  {
    std::cout << path << " loaded.\n";
    TriangleVisitor tv;
    customId = bh->addCustomObject( path, tv.getTriangles(), scale, pos, rot, phys );
    if (customId == -1) {
        model->accept(tv);
        //std::cout << "Num Triangles: " << tv.getTriangles()->size() << std::endl;
        customId = bh->addCustomObject( path, tv.getTriangles(), scale, pos, rot, phys, restitution );
    }
   
    // the matrix transform that holds the object picking needs to be set no the object else duplicate objects can not have a separate state 
    if (!pickable) 
        mt->setNodeMask(~2);

    m_physid.push_back( customId );
    m_objects.push_back( mt );
    MatrixTransform* centered = new MatrixTransform;
    centered->setMatrix( Matrixd::scale(scale,scale,scale) );
    mt->addChild(centered);
    centered->addChild(model);
    //mt->addChild(model);
    m_scales.push_back(scale);
    numObjects++;
  } 
  else 
  {
    std::cout << path << " could not be loaded.\n";
  }
  
  return mt;
}

MatrixTransform* ObjectFactory::addHollowBox( Vec3 pos, Vec3 halfLengths, bool phys, bool render ) {
  MatrixTransform* mt = new MatrixTransform;
  
  numObjects++;
  m_objects.push_back( mt );
  m_scales.push_back(1.0f);
  m_physid.push_back( bh->addHollowBox( pos, halfLengths, phys ) );
  
  return mt;
}

MatrixTransform* ObjectFactory::addAntiGravityField( Vec3 pos, Vec3 halfLengths, Vec3 grav, bool phys ) {
  MatrixTransform* mt = new MatrixTransform;
  
  if( phys )
  {
    Geode * box = new Geode;
    Box * boxprim = new Box( Vec3(0,0,0), 1,1,1);
    boxprim->setHalfLengths( halfLengths );
    ShapeDrawable * sd = new ShapeDrawable(boxprim);
    box->addDrawable(sd);
    Matrix boxm;
    boxm.makeTranslate(pos);
    mt->setMatrix( boxm );
    mt->addChild( box );
    StateSet* ss = box->getOrCreateStateSet();
    PolygonMode* pg = new PolygonMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
    ss->setAttributeAndModes(pg, StateAttribute::ON | StateAttribute::OVERRIDE);
    box->setNodeMask(~2);   
  }
    
  bh->addAntiGravityField( pos, halfLengths, grav );
  numObjects++;
  m_objects.push_back( mt );
  m_scales.push_back(1.0f);
  m_physid.push_back( -1 );
  
  return mt;
}

MatrixTransform* ObjectFactory::addPlane( Vec3 pos, double halfLength, Vec3 normal, bool phys, bool render ) {
    MatrixTransform* mt;
    
    Geode * floor = new Geode;
    Geometry* floorGeometry = new Geometry;
    floor->addDrawable( floorGeometry );
    
    const float floorWidth = 1000.0f;
    Vec3Array* floorVerts = new Vec3Array;
    floorVerts->push_back( Vec3(-floorWidth, -floorWidth, 0.0f) );
    floorVerts->push_back( Vec3(floorWidth, -floorWidth, 0.0f) );
    floorVerts->push_back( Vec3(floorWidth, floorWidth, 0.0f) );
    floorVerts->push_back( Vec3(-floorWidth, floorWidth, 0.0f) );
    //floorVerts->push_back( Vec3(0, 0, -floorWidth) );
    //floorVerts->push_back( Vec3(floorWidth, 0, -floorWidth) );
    //floorVerts->push_back( Vec3(floorWidth, 0, floorWidth) );
    //floorVerts->push_back( Vec3(0, 0, floorWidth) );
    floorGeometry->setVertexArray( floorVerts );
    
    Vec3Array * floorNorms = new Vec3Array;
    floorNorms->push_back( Vec3(0,0,1) );
    floorGeometry->setNormalArray( floorNorms );
    
    TemplateIndexArray<unsigned int, Array::UIntArrayType, 24, 4> *normalIndexArray;
    normalIndexArray =  new TemplateIndexArray<unsigned int, Array::UIntArrayType, 24, 4>();
    normalIndexArray->push_back(0);
    normalIndexArray->push_back(0);
    normalIndexArray->push_back(0);
    normalIndexArray->push_back(0);
    //floorGeometry->setNormalIndices(normalIndexArray);
    
    DrawElementsUInt* floorFace = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
    floorFace->push_back(3);
    floorFace->push_back(2);
    floorFace->push_back(1);
    floorFace->push_back(0);
    floorGeometry->addPrimitiveSet(floorFace);
    
    Vec4Array* colors = new Vec4Array;
    colors->push_back( Vec4(0.0f, 1.0f, 0.0f, 1.0f) );
    colors->push_back( Vec4(0.0f, 1.0f, 0.0f, 1.0f) );
    colors->push_back( Vec4(0.0f, 1.0f, 0.0f, 1.0f) );
    colors->push_back( Vec4(0.0f, 1.0f, 0.0f, 1.0f) );
    floorGeometry->setColorArray(colors);
    floorGeometry->setColorBinding(Geometry::BIND_PER_VERTEX);
    
    return mt;
}

PositionAttitudeTransform* ObjectFactory::addLight( Vec3 pos, Vec4 diffuse, Vec4 specular, Vec4 ambient, StateSet* lightSS ) {
  LightSource* lightsrc = new LightSource;
  PositionAttitudeTransform* lightMat = new PositionAttitudeTransform();
  Light * light0 = new Light();
  
  lightMat->addChild( lightsrc );
  light0->setPosition( Vec4(0.0f, 0.0f, 0.0f, 1.0f) );
  light0->setLightNum(numLights++);
  light0->setAmbient( ambient );
  light0->setDiffuse( diffuse );
  light0->setSpecular( specular );
  light0->setConstantAttenuation(2.0f);
  
  lightsrc->setLight( light0 );
  lightsrc->setLocalStateSetModes(osg::StateAttribute::ON);
  lightsrc->setStateSetModes(*lightSS, osg::StateAttribute::ON);
  
  Geode * sphere = new Geode;
  Sphere * sphereprim = new Sphere( Vec3() , 50);
  ShapeDrawable * sdp = new ShapeDrawable(sphereprim);
  sphere->addDrawable(sdp);
  Material *material = new Material();
  material->setDiffuse(Material::FRONT,  Vec4(0.0, 0.0, 0.0, 1.0));
  material->setEmission(Material::FRONT, diffuse);
  sphere->getOrCreateStateSet()->setAttribute(material);
  
  lightMat->addChild( sphere );
  lightMat->setPosition( pos );
  lightMat->setScale(Vec3(0.1,0.1,0.1));
  
  return lightMat;
}

void ObjectFactory::stepSim( double elapsedTime ) {
    bh->stepSim( elapsedTime );
    
    Matrixd m;
    for (int i = 0; i < numObjects; ++i) {
      if (m_physid[i] > -1 && m_physid[i] != grabbedPhysId) {
        bh->getWorldTransform( m_physid[i], m );
        //m.preMultScale(Vec3(m_scales[i], m_scales[i], m_scales[i]));
        //m.setTrans(pos);
        m_objects[i]->setMatrix( m );
      }
    }
    
    // only check once
    if (!m_wonGame) {
      int goalCount = 0;
      for (int i = 0; i < m_solvers.size(); ++i) {
        if (goalBounds != NULL && goalBounds->contains(m_solvers[i]->getMatrix().getTrans())) goalCount++;
      }
      if (goalCount > 0) m_wonGame = true;
    }
}

BulletHandler* ObjectFactory::getBulletHandler() {
  return bh;
}

MatrixTransform* ObjectFactory::addBoxHand( Vec3 halfLengths, Vec4 color ) {
  MatrixTransform* mt = new MatrixTransform;
  
  Geode * box = new Geode;
  Box * boxprim = new Box( Vec3(0,0,0), 1);
  boxprim->setHalfLengths( halfLengths );
  ShapeDrawable * sd = new ShapeDrawable(boxprim);
  sd->setColor( color );
  box->addDrawable(sd);
  //mt->addChild( box );
  
  handId = bh->addBox( Vec3(0,0,0), halfLengths, Quat(0,0,0,1), false );
  
  numObjects++;
  m_objects.push_back( mt );
  m_scales.push_back(1.0f);
  m_physid.push_back( handId );
  
  return mt;
}

MatrixTransform* ObjectFactory::addCylinderHand( double radius, double height, Vec4 color ) {
  MatrixTransform* mt = new MatrixTransform;
  Geode * tcyl = new Geode;
  Cylinder * tcylprim = new Cylinder( Vec3(0,0,0), radius, height);
  tcylprim->setRotation(Quat(3.14f / (float) 2, Vec3f(-1,0,0)));
  ShapeDrawable * cyld = new ShapeDrawable(tcylprim);
  tcyl->getOrCreateStateSet()->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
  cyld->setColor( color );
  tcyl->addDrawable(cyld);
  tcyl->setNodeMask(~2);
  //mt->addChild(tcyl);
  
  //bh->addHand( Vec3(0,0,0), Vec3(radius, 0, height/2) );
  /*handId = bh->addCylinder( Vec3(0,0,0), Vec3(radius, 0, height/2), false );
  
  numObjects++;
  m_objects.push_back( mt );
  m_physid.push_back( handId );
  */
  handMat = mt;
  return mt;
}

void ObjectFactory::updateHand( Matrixd & m, const Matrixd & cam ) {
  if (grabbedMatrix) {
    Matrixd mat = grabbedMatrixOffset * m;
    mat = mat * Matrixd::inverse(cam);
    grabbedMatrix->setMatrix( mat );
  }
}

void ObjectFactory::updateButtonState( int bs ) {
  bh->updateButtonState( bs );
}

bool ObjectFactory::grabObject( Matrixd & stylus, Node* root ) {
  Vec3d pointerEnd(0.f,1000000.f,0.f);
  pointerEnd = pointerEnd * stylus;
  
  osgUtil::IntersectVisitor objFinder;
  objFinder.setTraversalMask(2);
  LineSegment* pointerLine = new LineSegment();
  pointerLine->set( stylus.getTrans(), pointerEnd );
  objFinder.addLineSegment( pointerLine );
  root->accept( objFinder );
  
  osgUtil::IntersectVisitor::HitList hl = objFinder.getHitList(pointerLine);
  if (hl.empty()) return false;
  
  osgUtil::Hit closest = hl.front();
  std::string className = closest.getDrawable()->className();
  //std::cout << className << std::endl;
  
  NodePath np = closest.getNodePath();
  NodePath::reverse_iterator it;
  for (int i = 0; i < numObjects; ++i) {
    for (it = np.rbegin(); it != np.rend(); ++it) {
      if (m_objects[i] == *it ) {
        grabbedPhysId = m_physid[i];
        grabbedId = i;
        grabbedMatrix = (MatrixTransform*) (*it)->asTransform();
        break;
      }
    }
    if (grabbedPhysId != -1) break;
  }
  
  if (grabbedPhysId == -1) {
    grabbedMatrix = (MatrixTransform*) 0;
    return false;
  } else {
    // Put physics object away
    Matrixd garbage = Matrixd::translate(2000.,2000.,0.);
    bh->setWorldTransform(grabbedPhysId, garbage);
    
    // grab the shape if we alter its color
    grabbedShape = closest.getDrawable();
    
    // offset matrix
    grabbedMatrixOffset = grabbedMatrix->getMatrix();
    
    // convert it to world space
    ++it;
    for (; it != np.rend(); ++it)
      if ((*it)->asTransform() != NULL) {
      //std::cout << "Matrix Step: " << (*it)->asTransform()->asMatrixTransform()->getMatrix();
        grabbedMatrixOffset *= (*it)->asTransform()->asMatrixTransform()->getMatrix();
      }
    // make a "delta" matrix (final - initial)
    grabbedMatrixOffset = grabbedMatrixOffset * Matrixd::inverse(stylus);
    
    // highlight if a colored shapedrawable
    if (std::string(grabbedShape->className()).compare("ShapeDrawable") == 0) {
      grabbedIsSD = true;
      grabbedColor = ((ShapeDrawable*) grabbedShape)->getColor();
      ((ShapeDrawable*) grabbedShape)->setColor( Vec4(grabbedColor.r() + 0.4, grabbedColor.g() + 0.4, grabbedColor.b() + 0.4, 1) );
    } else {
      grabbedIsSD = false;
    }
  }
  return true;
}

void ObjectFactory::releaseObject() {
  if (grabbedPhysId != -1) {
    bh->setLinearVelocity(grabbedPhysId, Vec3(0,0,0));
    bh->activate(grabbedPhysId);
    if (grabbedIsSD) ((ShapeDrawable*) grabbedShape)->setColor(grabbedColor);
    Matrixd m = grabbedMatrix->getMatrix();
    bh->setWorldTransform( grabbedPhysId, m );
  }
  grabbedMatrix = 0;
  grabbedShape = 0;
  grabbedPhysId = grabbedId = -1;
  grabbedIsSD = false;
}

void ObjectFactory::addGoalZone( Vec3 pos, Vec3 halfLengths ) {
  goalBounds = new BoundingBoxd();
  goalBounds->set(pos - halfLengths, pos + halfLengths);
}

bool ObjectFactory::wonGame() {
  return m_wonGame;
}

void ObjectFactory::resetGame() {
  m_wonGame = false;
}

Geometry* makeQuad( Vec3 center, Vec3 relX, Vec3 relY) {
    osg::Geometry* geo = new osg::Geometry();
    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(center+relX-relY);
    verts->push_back(center-relX-relY);
    verts->push_back(center-relX+relY);
    verts->push_back(center+relX+relY);

    geo->setVertexArray(verts);

    osg::DrawElementsUInt * ele = new osg::DrawElementsUInt(
            osg::PrimitiveSet::QUADS,0);

    ele->push_back(0);
    ele->push_back(1);
    ele->push_back(2);
    ele->push_back(3);
    geo->addPrimitiveSet(ele);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(osg::Vec4(1.0,1.0,1.0,1.0));

    geo->setColorArray(colors);
    geo->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::Vec2Array* texcoords = new osg::Vec2Array;
    texcoords->push_back(osg::Vec2(1,0));
    texcoords->push_back(osg::Vec2(0,0));
    texcoords->push_back(osg::Vec2(0,1));
    texcoords->push_back(osg::Vec2(1,1));
    geo->setTexCoordArray(0,texcoords);
    
    return geo;
}

// Tip: Next time implement with createTexturedQuadGeometry
Group* ObjectFactory::addSkybox( std::string folder ) {
    Group* g = new Group();
    Geode* geode_sides[6];
    for (int i = 0; i < 6; ++i) {
      geode_sides[i] = new Geode();
      g->addChild(geode_sides[i]);
    }
    Geometry* geometry_sides[6];
    geometry_sides[0] = makeQuad(Vec3(0.5f,0,0), Vec3(0,-0.501f,0), Vec3(0,0,0.501f)); // posx
    geometry_sides[1] = makeQuad(Vec3(-0.5f,0,0), Vec3(0,0.501f,0), Vec3(0,0,0.501f));// negx
    geometry_sides[2] = makeQuad(Vec3(0,0.5f,0), Vec3(0.501f,0,0), Vec3(0,0,0.501f)); // posy
    geometry_sides[3] = makeQuad(Vec3(0,-0.5f,0), Vec3(-0.501f,0,0), Vec3(0,0,0.501f));// negy
    geometry_sides[4] = makeQuad(Vec3(0,0,0.5f), Vec3(0,-0.501f,0), Vec3(-0.501f,0,0)); // posz
    geometry_sides[5] = makeQuad(Vec3(0,0,-0.5f), Vec3(0,0.501f,0), Vec3(-0.501f,0,0));// negz
    
    const std::string file_names[] = { "posx", "negx", "posy", "negy", "posz", "negz" };
    const std::string file_ext = ".jpg";
    for (int i = 0; i < 6; ++i) {
      geode_sides[i]->addDrawable(geometry_sides[i]);
    
      Texture2D* tex = new osg::Texture2D();
      Image* image = osgDB::readImageFile(folder + file_names[i] + file_ext);
      if (!image) {
        cout << "Failed to load " << file_names[i] << endl;
        return g;
      }
      tex->setImage(image);

      osg::StateSet * stateset = geode_sides[i]->getOrCreateStateSet();
      stateset->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
      stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    }
    
    return g;
}

void ObjectFactory::setWorldTransform(MatrixTransform* node, Matrixd & ntrans) {
  for (int i = 0; i < numObjects; ++i) {
      if (m_objects[i] == node ) {
        bh->setWorldTransform(m_physid[i], ntrans);
        break;
      }
  }
}

void ObjectFactory::setLinearVelocity(MatrixTransform* node, Vec3 vel) {
  for (int i = 0; i < numObjects; ++i) {
      if (m_objects[i] == node ) {
        bh->setLinearVelocity(m_physid[i], vel);
        break;
      }
  }
}
