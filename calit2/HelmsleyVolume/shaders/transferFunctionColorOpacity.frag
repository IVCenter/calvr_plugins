#version 460

#pragma import_defines ( COLOR_FUNCTION )



uniform float OpacityCenter[10];
uniform float OpacityWidth[10];
uniform float OpacityTopWidth[10];
uniform float OpacityMult[10];
uniform float Lowest[10];
uniform float TriangleCount = 1.0;

uniform float ContrastBottom;
uniform float ContrastTop;
uniform float Brightness;

out vec4 FragColor;

in vs_out {
	vec2 uv;
} i;

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 ra = i.uv;
	vec4 col = vec4(0.0,0.0,0.0,0.0);

	float highestOpacity = 0.0;
	for(int i = 0; i < TriangleCount; i++){
		col.a = smoothstep((OpacityCenter[i]-OpacityTopWidth[i]) - (OpacityWidth[i] - OpacityTopWidth[i]), OpacityCenter[i] - OpacityTopWidth[i], ra.r);
		if(col.a == 1.0){
			col.a = 1.0 - smoothstep(OpacityCenter[i]+OpacityTopWidth[i], OpacityCenter[i]+ OpacityTopWidth[i] + (OpacityWidth[i] - OpacityTopWidth[i]), ra.r);
		}
		if(col.a != 0.0){
			col.a *= OpacityMult[i];
			float lowestLimit = Lowest[i];
			col.a = max(col.a, lowestLimit);
			if(col.a >= highestOpacity){
				highestOpacity = col.a;
			}
		}
	}
	col.a = highestOpacity;
   
	ra.r = (ra.r - ContrastBottom) / (ContrastTop - ContrastBottom);
	ra.r = max(0, min(1, ra.r));
	ra.r = clamp(ra.r + (Brightness - 0.5), 0.0, 1.0);

    #ifdef COLOR_FUNCTION
		col.rgb = COLOR_FUNCTION
	#else
		col.rgb = ra.rrr;
	#endif

	

	
	FragColor = vec4(col);
}