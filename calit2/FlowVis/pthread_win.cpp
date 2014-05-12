#include "pthread_win.h"

#pragma comment(lib, "Ws2_32.lib")

int pthread_mutex_lock(HANDLE * mutex)
{ 
	if(*mutex == NULL)
	{ 
		HANDLE p = CreateMutex(NULL, FALSE, NULL);
		if(InterlockedCompareExchangePointer((PVOID*)mutex, (PVOID)p, NULL) != NULL)
		{
			CloseHandle(p);
		}
	}
	return WaitForSingleObject(*mutex, INFINITE) == WAIT_FAILED;
}