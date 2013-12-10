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

#ifndef PTCLDEMO_PTCL_EMITTER
#define PTCLDEMO_PTCL_EMITTER 1

#include <osg/ref_ptr>
#include <osgCompute/Module>
#include <osgCompute/Memory>

namespace PtclDemo
{
    /**
    */
    class EmitterBox : public osgCompute::Resource
    {
    public:
        EmitterBox() : osgCompute::Resource() {}

        META_Object(PtclDemo,EmitterBox)

        osg::Vec3f _min;
        osg::Vec3f _max;

    protected:
        virtual ~EmitterBox() {}

    private:
        EmitterBox(const EmitterBox&, const osg::CopyOp& ) {} 
        inline EmitterBox &operator=(const EmitterBox &) { return *this; }
    };
        
    /**
    */
    class PtclEmitter : public osgCompute::Module 
    {
    public:
        PtclEmitter() : osgCompute::Module() {clearLocal();}

        META_Object( PtclDemo, PtclEmitter )

        virtual bool init();
        virtual void launch();
        virtual void acceptResource( osgCompute::Resource& resource );

        virtual void clear() { clearLocal(); osgCompute::Module::clear(); }
    protected:
        virtual ~PtclEmitter();
        void clearLocal();

        unsigned int                                      _numBlocks;
        unsigned int                                      _numThreads;
    
        osg::ref_ptr<EmitterBox>                          _box;
        osg::ref_ptr<osgCompute::Memory>                  _ptcls;
        osg::ref_ptr<osgCompute::Memory>                  _seeds;

    private:
        PtclEmitter(const PtclEmitter&, const osg::CopyOp& ) {} 
        inline PtclEmitter &operator=(const PtclEmitter &) { return *this; }
    };

};

#endif // PTCLDEMO_PTCL_EMITTER
