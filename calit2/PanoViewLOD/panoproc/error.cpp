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

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

//------------------------------------------------------------------------------

static const char *executable = NULL;

void set_executable(const char *str)
{
    executable = str;
}

//------------------------------------------------------------------------------

static void do_error(va_list ap, const char *fmt, const char *tag, int no)
{
    if (executable)
        fprintf(stderr, "%s : ", executable);

    if (tag)
        fprintf(stderr, "%s : ", tag);

    if (fmt)
       vfprintf(stderr, fmt, ap);

    if (no)
        fprintf(stderr, " : %s\n", strerror(no));
    else
        fprintf(stderr, "\n");
}

//------------------------------------------------------------------------------

void app_err_exit(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    do_error(ap, fmt, "Error", 0);
    va_end(ap);
    abort();
}

void sys_err_exit(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    do_error(ap, fmt, "Error", errno);
    va_end(ap);
    abort();
}

//------------------------------------------------------------------------------

void app_err_mesg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    do_error(ap, fmt, "Error", 0);
    va_end(ap);
}

void sys_err_mesg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    do_error(ap, fmt, "Error", errno);
    va_end(ap);
}

void app_wrn_mesg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    do_error(ap, fmt, "Warning", 0);
    va_end(ap);
}

void sys_wrn_mesg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    do_error(ap, fmt, "Warning", errno);
    va_end(ap);
}

void app_log_mesg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    do_error(ap, fmt, 0, 0);
    va_end(ap);
}

void sys_log_mesg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    do_error(ap, fmt, 0, errno);
    va_end(ap);
}

//void *app_err_null(const char *fmt, ...)
//{
//    va_list ap;
//    va_start(ap, fmt);
//    do_error(ap, fmt, 0);
//    va_end(ap);
//    return NULL;
//}
//
//void *sys_err_null(const char *fmt, ...)
//{
//    va_list ap;
//    va_start(ap, fmt);
//    do_error(ap, fmt, errno);
//    va_end(ap);
//    return NULL;
//}
//
//size_t app_err_zero(const char *fmt, ...)
//{
//    va_list ap;
//    va_start(ap, fmt);
//    do_error(ap, fmt, 0);
//    va_end(ap);
//    return 0;
//}
//
//size_t sys_err_zero(const char *fmt, ...)
//{
//    va_list ap;
//    va_start(ap, fmt);
//    do_error(ap, fmt, errno);
//    va_end(ap);
//    return 0;
//}

//------------------------------------------------------------------------------