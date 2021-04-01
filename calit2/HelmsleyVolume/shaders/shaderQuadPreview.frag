#version 460

#pragma import_defines ( COLOR_FUNCTION )

uniform float Center;
uniform float Width;
uniform float TopWidth;
uniform float OpacityMult;
uniform float Lowest;

uniform float ContrastBottom;
uniform float ContrastTop;
uniform float Brightness;

uniform vec3 leftColor = vec3(1,0,0);
uniform vec3 rightColor = vec3(1,1,1);

out vec4 FragColor;

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

void main() {
    vec2 ra = i.uv;

  

	ra.r = mix(Center-(Width/2), Center+(Width/2), ra.r);
	ra.r = (ra.r - ContrastBottom) / (ContrastTop - ContrastBottom);
	ra.r = max(0, min(1, ra.r));
	ra.r = clamp(ra.r , 0.0, 1.0);
    vec3 col = vec3(0.0,0.0,0.0);



//    #ifdef COLOR_FUNCTION
//		col = COLOR_FUNCTION
//	#else
//		col = vec3(ra.r, ra.r, ra.r);
//	#endif
col = COLOR_FUNCTION

	
	FragColor = vec4(col, 1);
}