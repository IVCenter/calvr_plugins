#ifndef PTHREAD_WIN_H
#define PTHREAD_WIN_H

#include <Windows.h>

#define pthread_mutex_t HANDLE
#define pthread_t HANDLE

#define pthread_mutex_init(a,b) *a = CreateMutex(NULL,FALSE,NULL)
#define pthread_mutex_destroy(a) CloseHandle(*a)
//#define pthread_mutex_lock(a) WaitForSingleObject(*a,INFINITE)
#define pthread_mutex_unlock(a) ReleaseMutex(*a)
#define pthread_create(a,b,c,d) *a = CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)c,d,0,NULL)
#define pthread_join(a,b) WaitForMultipleObjects(1,&a, TRUE, INFINITE);
#define PTHREAD_MUTEX_INITIALIZER NULL

int pthread_mutex_lock(HANDLE * mutex);

#endif