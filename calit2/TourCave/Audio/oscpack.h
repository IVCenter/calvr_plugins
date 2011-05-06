#ifndef __OSC_PACK_H__
#define __OSC_PACK_H__
/*
 *  oscpack.h
 *
 *  Created by Toshiro Yamada on 04/01/10.
 *  Copyright 2010 Calit2, UCSD. All rights reserved.
 *
 */

#include <stdint.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif
	
/*	
 *	oscpack() can be used to serialize data into a format specified by 
 *	OpenSoundControl specification 1.0 (http://opensoundcontrol.org/spec-1_0).
 *
 *	NOTE: oscpack() does NOT allocate memory for the argument buf and will not
 *	check size of buf. It is the responsibility of the caller to ensure that
 *	buf is valid and is sufficiently large. Caller may use oscsize() to get the
 *	exact size of the OSC message.
 *
 *	About OpenSoundControl packet:
 *	"An OSC packet consists of its contents, a contiguous block of binary data,
 *	and its size, the number of 8-bit bytes that comprise the contents. The
 *	size of an OSC packet is always a multiple of 4." from OSC specification.
 *
 *
 *	Arguments:
 *		uint8_t* buf: Destination of the osc message buffer. The size of the 
 *					  buffer should be sufficiently large.
 *		char* addr: OSC Address. Must start with '/' and match the syntax of
 *					URLs.
 *		char* format: See below for available formats.
 *
 *	Supported formats:
 *		i: 32-bit integer				h: 64-bit integer
 *		f: 32-bit floating point		d: 64-bit double floating point
 *		s: string (array of char)		c: ASCII character
 *		T: True  (no argument needed)	F: False (no argument needed)
 *		N: Nil (no argument needed)		I: Infinitum (no argument needed)
 *
 *	Unsupported formats (2010-06-20):
 *		b: blob							t: timetag
 *		r: 32 bit RGBA color (array of 4 32-bit integer?)
 *		m: 4 byte MIDI message. Bytes from MSB to LSB are: 
 *		   port id, status byte, data1, data2
 *		
 *
 *	Return:
 *		Size of the OpenSoundControl data.
 *
 *
 *	Usage example for UDP:
 *		uint8_t packet[256];
 *		int32_t size;
 *		size = oscpack(packet, "/osc/address", "ifs", 123, 1.23, "osc message");
 *		send(socket, packet, size, 0); // use UDP socket
 *
 *	Usage example for TCP:
 *		// For sending OSC over TCP, the first 4 bytes need to specify the size
 *		// of the OSC message.
 *		uint8_t packet[256];
 *		int32_t size, bit32;
 *		size = oscpack(packet+4, "/osc/address", "ifs", 123, 1.23, "osc message");
 *		bit32 = htonl(size);
 *		memcpy(packet, &bit32, 4);
 *		send(socket, packet, size+4, 0); // use TCP socket
 */

int32_t oscpack(uint8_t* buf, const char* addr, const char* format, ...);

/* Real implementation of oscpack */
int32_t voscpack(uint8_t* buf, const char* addr, const char* format, va_list arg);
	
/*
 *	oscsize can be used to calculate the size of OpenSoundControl message. 
 *	See oscpack for usage.
 *	
 */

int32_t oscsize(const char* addr, const char* format, ...);

#ifdef __cplusplus
}
#endif

#endif // __OSC_PACK_H__
