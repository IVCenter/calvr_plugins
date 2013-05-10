//  Copyright (C) 2005-2011 Robert Kooima
//
//  THUMB is free software; you can redistribute it and/or modify it under
//  the terms of  the GNU General Public License as  published by the Free
//  Software  Foundation;  either version 2  of the  License,  or (at your
//  option) any later version.
//
//  This program  is distributed in the  hope that it will  be useful, but
//  WITHOUT   ANY  WARRANTY;   without  even   the  implied   warranty  of
//  MERCHANTABILITY  or FITNESS  FOR A  PARTICULAR PURPOSE.   See  the GNU
//  General Public License for more details.

#include <cstdlib>
#include <cassert>
#include <sstream>
#include <iostream>
#include <unistd.h>

#include "sph-cache.hpp"
#include "cube.hpp"

//------------------------------------------------------------------------------

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <OpenThreads/ScopedLock>
#include <cvrConfig/ConfigManager.h>

//#define CACHE_PRINT_DEBUG

DiskCache * sph_cache::_diskCache = NULL;

static bool exists(const std::string& name)
{
    struct stat info;

    if (stat(name.c_str(), &info) == 0)
        return ((info.st_mode & S_IFMT) == S_IFREG);
    else
        return false;
}

//------------------------------------------------------------------------------

static void clear(GLuint t)
{
    static const GLfloat p[] = { 0, 0, 0, 0 };
        
    glBindTexture  (GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, p);
    glTexImage2D   (GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_FLOAT, p);
}

// Select an OpenGL internal texture format for an image with c channels and
// b bits per channel.

static GLenum internal_form(uint16 c, uint16 b)
{
    if (b == 8)
        switch (c)
        {
        case  1: return GL_LUMINANCE8;
        case  2: return GL_LUMINANCE_ALPHA;
        case  3: return GL_RGBA8; // *
        default: return GL_RGBA8;
        }
    if (b == 16)
        switch (c)
        {
        case  1: return GL_LUMINANCE16;
        case  2: return GL_LUMINANCE16_ALPHA16;
        case  3: return GL_RGB16;
        default: return GL_RGBA16;
        }
    if (b == 32)
        switch (c)
        {
        case  1: return GL_R32F;
        case  2: return GL_RG32F;
        case  3: return GL_RGB32F;
        default: return GL_RGBA32F;
        }
        
    return 0;
}

// Select an OpenGL external texture format for an image with c channels.

static GLenum external_form(uint16 c, uint16 b)
{
    if (b == 8)
        switch (c)
        {
        case  1: return GL_LUMINANCE;
        case  2: return GL_LUMINANCE_ALPHA;
        case  3: return GL_BGRA; // *
        default: return GL_BGRA;
        }
    else
        switch (c)
        {
        case  1: return GL_LUMINANCE;
        case  2: return GL_LUMINANCE_ALPHA;
        case  3: return GL_RGB;
        default: return GL_RGBA;
        }
}

// Select an OpenGL data type for an image with c channels of b bits.

static GLenum external_type(uint16 c, uint16 b)
{
    if (b ==  8 && c == 3) return GL_UNSIGNED_INT_8_8_8_8_REV; // *
    if (b ==  8 && c == 4) return GL_UNSIGNED_INT_8_8_8_8_REV;

    if (b ==  8) return GL_UNSIGNED_BYTE;
    if (b == 16) return GL_UNSIGNED_SHORT;
    if (b == 32) return GL_FLOAT;

    return 0;
}


static void initTexture(GLuint o, uint32 w, uint32 h, uint16 c, uint16 b)
{
    static const GLfloat p[] = { 0, 0, 0, 0 };
        
    glBindTexture  (GL_TEXTURE_2D, o);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, p);
    glTexImage2D (GL_TEXTURE_2D, 0, internal_form(c, b), w, h, 1,
                                        external_form(c, b),
                                        external_type(c, b), NULL);
    //glBindTexture  (GL_TEXTURE_2D, 0);
}

// AP - Function that I was using to test pbo unmap timing with different options, should never be called in production.  Also it exits at the end.
void pbotiming(GLuint pbo)
{
    int size = 5000000;
    void * p;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    {
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_STREAM_DRAW);
	p = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, size, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
	if(!p)
	{
	    std::cerr << "Invalid map." << std::endl;
	}

	memset(p,0x00,size);

	struct timeval start, end;
	gettimeofday(&start,NULL);

	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

	gettimeofday(&end,NULL);

	std::cerr << "Full unmap time: " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec)/ 1000000.0) << std::endl;

	p = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, size, GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
	memset(p,0x01,size);

	float segments = 5.0;
	int ssize = (int)(size / segments);
	if(segments * ssize < size)
	{
	    ssize++;
	}
	int copied = 0;
	for(int i = 0; i < segments; i++)
	{
	    int copy = std::min(size - copied,ssize);
	    
	    std::cerr << "Offset: " << copied << " length: " << copy << std::endl;

	    struct timeval fstart, fend;

	    gettimeofday(&fstart,NULL);

	    glFlushMappedBufferRange(GL_PIXEL_UNPACK_BUFFER,copied,copy);
	    //glFinish();

	    gettimeofday(&fend,NULL);

	    std::cerr << "Segment flush time: " << (fend.tv_sec - fstart.tv_sec) + ((fend.tv_usec - fstart.tv_usec)/ 1000000.0) << std::endl;

	    copied += copy;
	    sleep(1);
	}

	gettimeofday(&start,NULL);

	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

	gettimeofday(&end,NULL);

	std::cerr << "Final unmap time: " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec)/ 1000000.0) << std::endl;

    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    exit(0);
}


// * 24-bit images are always padded to 32 bits.

//------------------------------------------------------------------------------

// Construct a load task. Map the PBO to provide a destination for the loader.

sph_task::sph_task(int f, int i, GLuint u, GLsizei s, sph_cache * c, int t) : sph_item(f, i), u(u), cache(c), timestamp(t), valid(true)
{
    //pbotiming(u);
    size = s;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, u);
    {
        glBufferData(GL_PIXEL_UNPACK_BUFFER, s, 0, GL_STREAM_DRAW);
        //p = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, size, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
	p = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	if(!p)
	{
	    std::cerr << "Invalid map." << std::endl;
	}
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// Upload the pixel buffer to a newly-generated OpenGL texture object.

bool sph_task::make_texture(GLuint o, uint32 w, uint32 h, uint16 c, uint16 b)
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, u);
    {
#ifdef CACHE_PRINT_DEBUG
	struct timeval ustart, uend, tstart, tend;
	gettimeofday(&ustart,NULL);
#endif

	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

#ifdef CACHE_PRINT_DEBUG
	gettimeofday(&uend,NULL);
#endif

	glBindTexture(GL_TEXTURE_2D, o);
	glTexImage2D (GL_TEXTURE_2D, 0, internal_form(c, b), w, h, 1,
	  external_form(c, b),
	  external_type(c, b), 0);

#ifdef CACHE_PRINT_DEBUG
	gettimeofday(&tstart,NULL);
#endif

	/*glTexSubImage2D (GL_TEXTURE_2D, 0, -1, -1, w, h,
		external_form(c, b),
		external_type(c, b), 0);*/

#ifdef CACHE_PRINT_DEBUG
	gettimeofday(&tend,NULL);
	std::cerr << "Size: " << size <<  " Unmap time: " << (uend.tv_sec - ustart.tv_sec) + ((uend.tv_usec - ustart.tv_usec)/ 1000000.0) << " Texture time: " << (tend.tv_sec - tstart.tv_sec) + ((tend.tv_usec - tstart.tv_usec)/ 1000000.0) << std::endl;
#endif
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    return true;
}

// A texture was loaded but is no longer necessary. Discard the pixel buffer.

void sph_task::dump_texture()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, u);
    {
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// Load the current TIFF directory into a newly-allocated pixel buffer.

void sph_task::load_texture(TIFF *T, uint32 w, uint32 h, uint16 c, uint16 b)
{
    // Confirm the page format.
    
    uint32 W, H;
    uint16 C, B;
    
    TIFFGetField(T, TIFFTAG_IMAGEWIDTH,      &W);
    TIFFGetField(T, TIFFTAG_IMAGELENGTH,     &H);
    TIFFGetField(T, TIFFTAG_BITSPERSAMPLE,   &B);
    TIFFGetField(T, TIFFTAG_SAMPLESPERPIXEL, &C);
    
    if (W == w && H == h && B == b && C == c)
    {
        // Pad a 24-bit image to 32-bit BGRA.

        if (c == 3 && b == 8)
        {
            if (void *q = malloc(TIFFScanlineSize(T)))
            {
                const uint32 S = w * 4 * b / 8;

                for (uint32 r = 0; r < h; ++r)
                {
                    TIFFReadScanline(T, q, r, 0);
                    
                    for (int j = w - 1; j >= 0; --j)
                    {
                        uint8 *s = (uint8 *) q         + j * c * b / 8;
                        uint8 *d = (uint8 *) p + r * S + j * 4 * b / 8;
                        
                        d[0] = s[2];
                        d[1] = s[1];
                        d[2] = s[0];
                        d[3] = 0xFF;
                    }
                }
                free(q);
            }
        }
        else
        {
            const uint32 S = (uint32) TIFFScanlineSize(T);

            for (uint32 r = 0; r < h; ++r)
                TIFFReadScanline(T, (uint8 *) p + r * S, r, 0);
        }
    }
}

// Construct a file table entry. Load the TIFF briefly to determine its format.
    
sph_file::sph_file(const std::string& tiff)
{
    // If the given file name is absolute, use it.
    
    if (exists(tiff))
        name = tiff;

    // Otherwise, search panorama path for the file.

    else if (char *val = getenv("PANOPATH"))
    {
        std::stringstream list(val);
        std::string       path;
        std::string       temp;
        
        while (std::getline(list, path, ':'))
        {
            temp = path + "/" + tiff;

            if (exists(temp))
            {
                name = temp;
                break;
            }
        }
    }

    if (!name.empty())
    {
	std::cerr << "File name: " << name << std::endl;
        if (TIFF *T = TIFFOpen(name.c_str(), "r"))
        {
            TIFFGetField(T, TIFFTAG_IMAGEWIDTH,      &w);
            TIFFGetField(T, TIFFTAG_IMAGELENGTH,     &h);
            TIFFGetField(T, TIFFTAG_BITSPERSAMPLE,   &b);
            TIFFGetField(T, TIFFTAG_SAMPLESPERPIXEL, &c);

            TIFFClose(T);
        }
    }
}

//------------------------------------------------------------------------------

sph_set::~sph_set()
{
    std::map<sph_page, int>::iterator i;
    
    for (i = m.begin(); i != m.end(); ++i)
        glDeleteTextures(1, &i->first.o);
}

void sph_set::insert(sph_page page, int t)
{
    m[page] = t;
}

void sph_set::remove(sph_page page)
{
    m.erase(page);
}

sph_page sph_set::search(sph_page page, int t)
{
    std::map<sph_page, int>::iterator i = m.find(page);
    
    if (i != m.end())
    {
        sph_page p = i->first;

        remove(p);
        insert(p, t);
        return(p);
    }
    return sph_page();
}

sph_page sph_set::eject(int t, int i)
{
    assert(!m.empty());
    
    // Determine the lowest priority and least-recently used pages.
    
    std::map<sph_page, int>::iterator a = m.end();
    std::map<sph_page, int>::iterator l = m.end();
    std::map<sph_page, int>::iterator e;
    
    for (e = m.begin(); e != m.end(); ++e)
//      if (e->first.i > 5) // HACK
        {
            if (a == m.end() || e->second < a->second) a = e;
                                                       l = e;
        }

    // If the LRU page was not used in this frame or the last, eject it.
    // Otherwise consider the lowest-priority loaded page and eject if it
    // has lower priority than the incoming page.

    if (a != m.end() && a->second < t - 2)
    {
        sph_page page = a->first;
        m.erase(a);
        return page;
    }
    if (l != m.end() && i < l->first.i)
    {
        sph_page page = l->first;
        m.erase(l);
        return page;
    }
    return sph_page();
}

void sph_set::draw()
{
    int l = log2(size);
    int r = (l & 1) ? (1 << ((l - 1) / 2)) : (1 << (l / 2));
    int c = (l & 1) ? (1 << ((l + 1) / 2)) : (1 << (l / 2));

    glUseProgram(0);

    glPushAttrib(GL_VIEWPORT_BIT | GL_ENABLE_BIT);
    {
        glDisable(GL_LIGHTING);
        glEnable(GL_TEXTURE_2D);

        glViewport(0, 0, c * 32, r * 32);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(c, 0, r, 0, 0, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glColor4f(1.f, 1.f, 1.f, 1.f);

        std::map<sph_page, int>::iterator i = m.begin();

        for     (int y = 0; y < r; ++y)
            for (int x = 0; x < c; ++x)

                if (i != m.end())
                {
                    glBindTexture(GL_TEXTURE_2D, i->first.o);
                    glBegin(GL_QUADS);
                    {
                        glTexCoord2f(0.f, 0.f); glVertex2i(x,     y);
                        glTexCoord2f(1.f, 0.f); glVertex2i(x + 1, y);
                        glTexCoord2f(1.f, 1.f); glVertex2i(x + 1, y + 1);
                        glTexCoord2f(0.f, 1.f); glVertex2i(x,     y + 1);
                    }
                    glEnd();
                    i++;
                }
                
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }
    glPopAttrib();
}

//------------------------------------------------------------------------------

sph_cache::sph_cache(int n) : pages(n), waits(n), needs(32), loads()
{
    GLuint b;
    int    i;
   
    static OpenThreads::Mutex _initMutex;
    {
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_initMutex);
	/*if(!_diskCache)
	{
	    std::cerr << "Warning: Creating DiskCache in sph_cache.  This may cause the DiskCache threads to bind to a single core." << std::endl;
	    _diskCache = new DiskCache(cvr::ConfigManager::getInt("value","Plugin.PanoViewLOD.DiskCacheSize",256));
	}*/
	if(!_diskCache)
	{
	    _useDiskCache = false;
	}
	else
	{
	    _useDiskCache = true;
	}
    }

    float targetFPS = cvr::ConfigManager::getFloat("value","Plugin.PanoViewLOD.TargetFPS",60.0);
    _maxTime = 1.0f / targetFPS;
    // give some time for other things
    _maxTime *= 0.9;
    
    // Launch the image loader threads.
    
    int loader(void *data);

    if(!_useDiskCache)
    {
	thread[0] = SDL_CreateThread(loader, this);
	thread[1] = SDL_CreateThread(loader, this);
	thread[2] = SDL_CreateThread(loader, this);
	thread[3] = SDL_CreateThread(loader, this);
    }

    // Generate pixel buffer objects.
 
    for (i = 0; i < 64; ++i)
    {
        glGenBuffers(1, &b);
        pbos.push_back(b);
    }

    // Initialize the default filler texture.
    
    glGenTextures(1, &filler);
    clear(filler);
}

sph_cache::~sph_cache()
{
    // Continue servicing the loads queue until the needs queue is emptied.
    
    while (!needs.empty())
        update(0);
    
    // Enqueue an exit command for each loader thread.
    
    if(!_useDiskCache)
    {
	needs.insert(sph_task(-1, -1));
	needs.insert(sph_task(-2, -2));
	needs.insert(sph_task(-3, -3));
	needs.insert(sph_task(-4, -4));

	// Await their exit. 

	int s;

	SDL_WaitThread(thread[0], &s);
	SDL_WaitThread(thread[1], &s);
	SDL_WaitThread(thread[2], &s);
	SDL_WaitThread(thread[3], &s);
    }

    // Release the pixel buffer objects.
    
    while (!pbos.empty())
    {
        glDeleteBuffers(1, &pbos.back());
        pbos.pop_back();
    }
    
    // Release the default texture.

    glDeleteTextures(1, &filler);
}

//------------------------------------------------------------------------------

static void debug_on(int l)
{
    static const GLfloat color[][3] = {
        { 1.0f, 0.0f, 0.0f },
        { 1.0f, 1.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 1.0f, 1.0f },
        { 0.0f, 0.0f, 1.0f },
        { 1.0f, 0.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f },
    };
    glPixelTransferf(GL_RED_SCALE,   color[l][0]);
    glPixelTransferf(GL_GREEN_SCALE, color[l][1]);
    glPixelTransferf(GL_BLUE_SCALE,  color[l][2]);                
}

//------------------------------------------------------------------------------

// Append a string to the file list and return its index. Queue the roots.

int sph_cache::add_file(const std::string& name)
{
    if(!_useDiskCache)
    {
	int f = int(files.size());

	files[f] = sph_file(name);

	return f;
    }
    else
    {
	int f = _diskCache->add_file(name);
	files[f] = sph_file(name);
	return f;
    }
}

// Return the texture object associated with the requested page. Request the
// image if necessary.

GLuint sph_cache::get_page(int f, int i, int t, int& n)
{
    // If this page is waiting, return the filler.

    sph_page wait = waits.search(sph_page(f, i), t);
    
    if (wait.valid())
    {
        n    = wait.t;
        return wait.o;
    }

    // If this page is loaded, return the texture.
    
    sph_page page = pages.search(sph_page(f, i), t);
    
    if (page.valid()) 
    {
        n    = page.t;
        return page.o;
    }
    
    // Otherwise request the page and add it to the waiting set.

    if (!needs.full() && !pbos.empty())
    {
        GLuint o = 0;
        
        if (pages.full())
            o = pages.eject(t, i).o;
        else
            glGenTextures(1, &o);
            
        if (o)
        {
	    if(!_useDiskCache)
	    {
		needs.insert(sph_task(f, i, pbos.deq(), pagelen(f), this, t));
	    }
	    else
	    {
		_diskCache->add_task(new sph_task(f, i, pbos.deq(), pagelen(f), this, t));
	    }
            waits.insert(sph_page(f, i, filler), t);
            pages.insert(sph_page(f, i, o),      t);            
            clear(o);
	    //initTexture(o, files[f].w, files[f].h,
	    //		    files[f].c, files[f].b);
        }
    }

    n = 0;
    return filler;
}

// Handle incoming textures on the loads queue.

void sph_cache::update(int t)
{
    struct timeval start, end;
    gettimeofday(&start,NULL);
    glPushAttrib(GL_PIXEL_MODE_BIT);
    {
	int c;
        for (c = 0; !loads.empty(); ++c)
        {
            sph_task task = loads.remove();
            sph_page page = pages.search(sph_page(task.f, task.i), t);
                            waits.remove(sph_page(task.f, task.i));

            if (page.valid())
            {
		if(task.valid)
		{
		    page.t = t;
		    pages.remove(page);
		    pages.insert(page, t);

		    if (debug)
			debug_on(face_level(task.i));

		    task.make_texture(page.o, files[task.f].w, files[task.f].h,
			    files[task.f].c, files[task.f].b);
		}
		else
		{
		    pages.remove(page);
		    task.dump_texture();
		    if(page.o)
		    {
			glDeleteTextures(1,&page.o);
			page.o = 0;
		    }
		}
            }
            else
	    {
                task.dump_texture();
	    }

            pbos.enq(task.u);

	    gettimeofday(&end,NULL);
	    float time = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec)/1000000.0);
	    if(time > _maxTime)
	    {
#ifdef CACHE_PRINT_DEBUG
		std::cerr << "Timeout break: Textures loaded: " << c+1 << std::endl;
#endif
		break;
	    }
        }

	/*if(c)
	{
	    std::cerr << "Textures loaded: " << c << std::endl;
	}*/
    }
    glPopAttrib();
}

void sph_cache::flush()
{
    while (!pages.empty())
        pages.eject(0, -1);
}

void sph_cache::draw()
{
    pages.draw();
}

//------------------------------------------------------------------------------

// Compute the length of a page buffer for file f.

GLsizei sph_cache::pagelen(int f)
{
    uint32 w = files[f].w;
    uint32 h = files[f].h;
    uint16 c = files[f].c;
    uint16 b = files[f].b;

    if (c == 3 && b == 8)
        return w * h * 4 * b / 8; // *
    else
        return w * h * c * b / 8;
}

//------------------------------------------------------------------------------

//static int up(TIFF *T, int i);
//static int dn(TIFF *T, int i);

// Seek upward to the root of the page tree and choose the appropriate base
// image. Navigate to the requested sub-image directory on the way back down.

int up(TIFF *T, int i)
{
    if (i < 6)
        return TIFFSetDirectory(T, i);
    else
    {
        if (up(T, face_parent(i)))
            return dn(T, face_index(i));
        else
            return 0;
    }
}

int dn(TIFF *T, int i)
{
    uint64_t *v;
    uint16  n;
    
    if (TIFFGetField(T, TIFFTAG_SUBIFD, &n, &v))
    {
        if (n > 0 && v[i] > 0)
            return TIFFSetSubDirectory(T, v[i]);
    }
    return 0;
}

// Load textures. Remove a task from the cache's needed queue, open and read
// the TIFF image file, and insert the task in the cache's loaded queue. Exit
// when given a negative file index.

int loader(void *data)
{
    sph_cache *cache = (sph_cache *) data;
    sph_task   task;
    
    while ((task = cache->needs.remove()).f >= 0)
    {
        if (!cache->files[task.f].name.empty())
        {
            if (TIFF *T = TIFFOpen(cache->files[task.f].name.c_str(), "r"))
            {
                if (up(T, task.i))
                {
                    uint32 w = cache->files[task.f].w;
                    uint32 h = cache->files[task.f].h;
                    uint16 c = cache->files[task.f].c;
                    uint16 b = cache->files[task.f].b;
                
                    task.load_texture(T, w, h, c, b);
                    cache->loads.insert(task);
                }
                TIFFClose(T);
            }
        }
    }
    return 0;
}

//------------------------------------------------------------------------------

