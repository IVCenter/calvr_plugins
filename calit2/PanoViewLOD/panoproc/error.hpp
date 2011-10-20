// Copyright (c) 2011 Robert Kooima
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#ifndef PANOPROC_ERROR_H
#define PANOPROC_ERROR_H

//------------------------------------------------------------------------------

void set_executable(const char *);

void   app_err_exit(const char *, ...);
void   sys_err_exit(const char *, ...);
void   app_err_mesg(const char *, ...);
void   sys_err_mesg(const char *, ...);
void   app_wrn_mesg(const char *, ...);
void   sys_wrn_mesg(const char *, ...);
void   app_log_mesg(const char *, ...);
void   sys_log_mesg(const char *, ...);

//void  *app_err_null(const char *, ...);
//void  *sys_err_null(const char *, ...);
//size_t app_err_zero(const char *, ...);
//size_t sys_err_zero(const char *, ...);

//------------------------------------------------------------------------------

#endif