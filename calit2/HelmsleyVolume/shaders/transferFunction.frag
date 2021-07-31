#version 460

#pragma import_defines ( COLOR_FUNCTION )

uniform float ContrastBottom;
uniform float ContrastTop;
uniform float Brightness;

out vec4 FragColor;

uniform vec3 leftColor = vec3(1,0,0);
uniform vec3 rightColor = vec3(1,1,1);

in vs_out {
	vec2 uv;
} i;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 custom(vec3 c) {
	vec3 color = mix(leftColor, rightColor, c);
	return color;
}


uint fireLut[256] = uint[256](0x000000, 0x060000, 0x0d0000, 0x120000, 0x160000, 0x190000, 0x1c0000, 0x1f0000, 0x220000, 0x240000, 0x260000, 0x280000, 0x2b0000, 0x2d0000, 0x2e0000, 0x300000, 0x320000, 0x340000, 0x350000, 0x370000, 0x380000, 0x3a0000, 0x3b0000, 0x3d0000, 0x3e0000, 0x400000, 0x410000, 0x430000, 0x440000, 0x460000, 0x470000, 0x490000, 0x4a0000, 0x4c0000, 0x4d0000, 0x4f0000, 0x500000, 0x520000, 0x530000, 0x550000, 0x560000, 0x580000, 0x590100, 0x5b0100, 0x5d0100, 0x5e0100, 0x600100, 0x610100, 0x630100, 0x650100, 0x660100, 0x680100, 0x690100, 0x6b0100, 0x6d0100, 0x6e0100, 0x700100, 0x710100, 0x730100, 0x750100, 0x760100, 0x780200, 0x7a0200, 0x7b0200, 0x7d0200, 0x7f0200, 0x800200, 0x820200, 0x840200, 0x850200, 0x870200, 0x890200, 0x8a0200, 0x8c0300, 0x8e0300, 0x900300, 0x910300, 0x930300, 0x950300, 0x960300, 0x980300, 0x9a0300, 0x9c0300, 0x9d0400, 0x9f0400, 0xa10400, 0xa20400, 0xa40400, 0xa60400, 0xa80400, 0xa90400, 0xab0500, 0xad0500, 0xaf0500, 0xb00500, 0xb20500, 0xb40500, 0xb60600, 0xb80600, 0xb90600, 0xbb0600, 0xbd0600, 0xbf0700, 0xc00700, 0xc20700, 0xc40700, 0xc60800, 0xc80800, 0xc90800, 0xcb0800, 0xcd0900, 0xcf0900, 0xd10900, 0xd20a00, 0xd40a00, 0xd60a00, 0xd80b00, 0xda0b00, 0xdb0c00, 0xdd0c00, 0xdf0d00, 0xe10d00, 0xe30e00, 0xe40f00, 0xe60f00, 0xe81000, 0xea1100, 0xeb1300, 0xed1400, 0xee1600, 0xf01800, 0xf11b00, 0xf21d00, 0xf32000, 0xf52300, 0xf62600, 0xf62900, 0xf72c00, 0xf82f00, 0xf93200, 0xf93500, 0xfa3800, 0xfa3b00, 0xfb3d00, 0xfb4000, 0xfb4300, 0xfc4600, 0xfc4900, 0xfc4b00, 0xfd4e00, 0xfd5100, 0xfd5300, 0xfd5600, 0xfd5800, 0xfe5b00, 0xfe5d00, 0xfe5f00, 0xfe6200, 0xfe6400, 0xfe6600, 0xfe6800, 0xfe6b00, 0xfe6d00, 0xfe6f00, 0xfe7100, 0xfe7300, 0xfe7500, 0xfe7700, 0xfe7900, 0xfe7c00, 0xff7e00, 0xff8000, 0xff8200, 0xff8300, 0xff8500, 0xff8700, 0xff8900, 0xff8b00, 0xff8d00, 0xff8f00, 0xff9100, 0xff9300, 0xff9400, 0xff9600, 0xff9800, 0xff9a00, 0xff9c00, 0xff9d00, 0xff9f00, 0xffa100, 0xffa300, 0xffa401, 0xffa601, 0xffa801, 0xffaa01, 0xffab01, 0xffad01, 0xffaf01, 0xffb001, 0xffb202, 0xffb402, 0xffb502, 0xffb702, 0xffb902, 0xffba02, 0xffbc03, 0xffbd03, 0xffbf03, 0xffc103, 0xffc204, 0xffc404, 0xffc604, 0xffc704, 0xffc905, 0xffca05, 0xffcc05, 0xffce06, 0xffcf06, 0xffd106, 0xffd207, 0xffd407, 0xffd508, 0xffd708, 0xffd909, 0xffda09, 0xffdc0a, 0xffdd0a, 0xffdf0b, 0xffe00b, 0xffe20c, 0xffe30d, 0xffe50e, 0xffe60f, 0xffe810, 0xffea11, 0xffeb12, 0xffed14, 0xffee17, 0xfff01a, 0xfff11e, 0xfff324, 0xfff42a, 0xfff532, 0xfff73b, 0xfff847, 0xfff953, 0xfffb62, 0xfffb72, 0xfffc83, 0xfffd95, 0xfffea8, 0xfffeba, 0xfffecc, 0xfffede, 0xfffeee, 0xffffff);
uint CET_18[256] = uint[256](0x000e5c,0x000f5e,0x000f60,0x011061,0x011063,0x011165,0x021166,0x031268,0x031269,0x04126b,0x05136c,0x07136e,0x081470,0x091471,0x0b1473,0x0d1574,0x0e1576,0x101677,0x121678,0x13167a,0x15177b,0x17177d,0x19177e,0x1b187f,0x1d1880,0x1f1882,0x211883,0x231984,0x251985,0x271986,0x291988,0x2b1a89,0x2d1a8a,0x2f1a8b,0x311a8c,0x341a8c,0x361a8d,0x391a8e,0x3b1a8f,0x3d1a8f,0x401a90,0x431a90,0x451a91,0x481a91,0x4b1991,0x4e1992,0x501992,0x531892,0x561892,0x591792,0x5b1792,0x5e1692,0x601692,0x631692,0x651592,0x681592,0x6a1492,0x6d1491,0x6f1391,0x711391,0x741291,0x761291,0x781191,0x7a1191,0x7c1090,0x7f1090,0x810f90,0x830f90,0x850e90,0x870e8f,0x890d8f,0x8b0d8f,0x8d0c8e,0x8f0c8e,0x910c8e,0x930b8d,0x950b8d,0x970b8d,0x990a8c,0x9b0a8c,0x9d0a8c,0x9f098b,0xa1098b,0xa3098a,0xa5098a,0xa7098a,0xa90989,0xaa0989,0xac0988,0xae0988,0xb00988,0xb20987,0xb30a87,0xb50a86,0xb70a86,0xb90b85,0xba0b85,0xbc0c85,0xbe0c84,0xbf0d84,0xc10d83,0xc30e83,0xc40e82,0xc60f82,0xc81081,0xc91181,0xcb1180,0xcd1280,0xce137f,0xd0147f,0xd1167e,0xd3177e,0xd4187d,0xd61a7c,0xd71b7c,0xd81d7b,0xda1e7a,0xdb2079,0xdc2179,0xde2378,0xdf2577,0xe02676,0xe12876,0xe22a75,0xe32c74,0xe42d73,0xe62f72,0xe73171,0xe83370,0xe9356f,0xe9366e,0xea386d,0xeb3a6c,0xec3c6b,0xed3e6a,0xee4069,0xee4267,0xef4466,0xf04565,0xf14764,0xf14963,0xf24b62,0xf24d61,0xf34f60,0xf3515f,0xf4525e,0xf4545d,0xf5565c,0xf5585b,0xf65a5a,0xf65c59,0xf65e58,0xf75f57,0xf76157,0xf76356,0xf86555,0xf86754,0xf86953,0xf86a52,0xf86c52,0xf86e51,0xf87050,0xf87250,0xf8744f,0xf8754e,0xf8774e,0xf8794d,0xf87b4c,0xf87d4c,0xf87e4b,0xf8804a,0xf8824a,0xf88349,0xf88548,0xf88748,0xf88847,0xf88a47,0xf88c46,0xf88d45,0xf98f45,0xf99044,0xf99243,0xf99343,0xf99542,0xf99641,0xf99841,0xf99940,0xf99b3f,0xf99c3f,0xfa9e3e,0xfa9f3d,0xfaa03d,0xfaa23c,0xfaa33b,0xfaa53b,0xfba63a,0xfba73a,0xfba939,0xfbaa39,0xfbab38,0xfbad38,0xfcae38,0xfcb037,0xfcb137,0xfcb237,0xfcb437,0xfcb537,0xfcb637,0xfcb837,0xfcb937,0xfcbb37,0xfcbc38,0xfcbd38,0xfcbf38,0xfcc038,0xfcc238,0xfcc339,0xfcc439,0xfcc639,0xfcc73a,0xfcc83a,0xfcca3a,0xfccb3b,0xfccd3b,0xfbce3b,0xfbcf3c,0xfbd13c,0xfbd23d,0xfbd33d,0xfbd53e,0xfbd63e,0xfbd83f,0xfad93f,0xfada40,0xfadc40,0xfadd41,0xfade41,0xf9e042,0xf9e142,0xf9e243,0xf9e444,0xf9e544,0xf8e745,0xf8e845,0xf8e946,0xf8eb47,0xf7ec47,0xf7ed48,0xf7ef48,0xf7f049,0xf6f14a,0xf6f34a,0xf6f44b,0xf5f64c,0xf5f74d,0xf5f84d);

vec3 useLut(float originalValue, int lutID){
	uint index = uint(originalValue * 255);
	uint hexColor; 
	if(lutID == 0)
		hexColor = fireLut[index];
	else if(lutID == 1)
		hexColor = CET_18[index];

	float rValue = float(hexColor / 256 / 256);
	float gValue = float(hexColor / 256 - int(rValue * 256.0));
	float bValue = float(hexColor - int(rValue * 256.0 * 256.0) - int(gValue * 256.0));
	return vec3(rValue / 255.0, gValue / 255.0, bValue / 255.0);
}

void main() {
    vec2 ra = i.uv;

    ra.r = (ra.r - ContrastBottom) / (ContrastTop - ContrastBottom);
	ra.r = max(0, min(1, ra.r));
	ra.r = clamp(ra.r, 0.0, 1.0);
    vec3 col = vec3(0.0,0.0,0.0);

    #ifdef COLOR_FUNCTION
		col = COLOR_FUNCTION
	#else
		col = vec3(ra.r);
	#endif

	
	FragColor = vec4(col, 1);
}