/* osgCompute - Copyright (C) 2008-2009 SVT Group
*
* This library is free software; you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of
* the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesse General Public License for more details.
*
* The full license is in LICENSE file included with this distribution.
*/

#include <vector_types.h>
#include <math.h>
#include <cstdlib>
#include <osg/Notify>
#include "PtclEmitter.h"

//------------------------------------------------------------------------------
extern "C"
void reseed( unsigned int numBlocks,
            unsigned int numThreads,
            void* ptcls,
            void* seeds,
            unsigned int seedCount,
            unsigned int seedIdx,
            float3 bbmin,
            float3 bbmax );


namespace PtclDemo
{
    //------------------------------------------------------------------------------
    PtclEmitter::~PtclEmitter()
    {
        clearLocal();
    }

    //------------------------------------------------------------------------------
    bool PtclEmitter::init()
    {
        if( !_ptcls.valid() || !_box.valid() )
        {
            osg::notify( osg::WARN )
                << "ParticleDemo::ParticleMover::init(): resources are missing."
                << std::endl;

            return false;
        }

        /////////////////////////
        // COMPUTE KERNEL SIZE //
        /////////////////////////
        // One Thread handles a single particle
        // buffer size must be a multiple of 128 x sizeof(float4)
        _numBlocks = _ptcls->getDimension(0) / 128;
        _numThreads = 128;

        return osgCompute::Module::init();
    }

    //------------------------------------------------------------------------------
    void PtclEmitter::launch()
    {
        if( isClear() )
            return;

		
		float3 bbmin = { _box->_min.x(), _box->_min.y(), _box->_min.z() };
		float3 bbmax = { _box->_max.x(), _box->_max.y(), _box->_max.z() };
		
        reseed(
            _numBlocks,
            _numThreads,
            _ptcls->map(),
            _seeds->map(),
            _seeds->getDimension(0),
            static_cast<unsigned int>(rand()),
            bbmin,
            bbmax );
    }

    //------------------------------------------------------------------------------
    void PtclEmitter::acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("PTCL_BUFFER") )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
        if( resource.isIdentifiedBy("PTCL_SEEDS") )
            _seeds = dynamic_cast<osgCompute::Memory*>( &resource );
        if( resource.isIdentifiedBy("EMITTER_BOX") )
            _box = dynamic_cast<EmitterBox*>( &resource );
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // PROTECTED FUNCTIONS ///////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //------------------------------------------------------------------------------
    void PtclEmitter::clearLocal()
    {
        _numBlocks = 1;
        _numThreads = 1;
        _ptcls = NULL;
        _seeds = NULL;
        _box = NULL;
    }
}
