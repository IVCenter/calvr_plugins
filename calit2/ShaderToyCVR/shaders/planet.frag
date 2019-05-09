const vec3 lightDirection = normalize(vec3(-.5, -.1, 0.0));
const vec3 betaR = vec3(.38, 1.35, 3.31);
const vec3 betaM = vec3(2.1);

#define ATMOSPHERE 1

#define innerRadius 1.0
#define atmoRadius 1.175
#define waterLevel .45

#define Hr .1332333
#define Hm .02
#define g .76

#define PI 3.14159265359
#define gamma 2.2
#define invgamma 1.0 / gamma

#define hash(a) fract(sin(a)*12345.0) 
float noise(vec3 x, float c1, float c2) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*c2+ c1*p.z;
    return mix(
        mix(
            mix(hash(n+0.0),hash(n+1.0),f.x),
            mix(hash(n+c2),hash(n+c2+1.0),f.x),
            f.y),
        mix(
            mix(hash(n+c1),hash(n+c1+1.0),f.x),
            mix(hash(n+c1+c2),hash(n+c1+c2+1.0),f.x),
            f.y),
        f.z);
}
float noise(vec3 p){

	float a = noise(p, 883.0, 971.0);
	float b = noise(p + 0.5, 113.0, 157.0);
	return (a + b) * 0.5;
}
float map4( in vec3 p ) {
	float f;
    f  = 0.50000*noise( p ); p = p*2.02;
    f += 0.25000*noise( p ); p = p*2.03;
    f += 0.12500*noise( p ); p = p*2.01;
    f += 0.06250*noise( p );
	return clamp(f, 0.0, 1.0);
}

float height(vec3 dir){
    float h;
    
    if (abs(dir.y) > innerRadius * .9){
        dir += 10.0;
        h = map4(dir * 3.0+ map4(dir * 5.0) * (sin(iTime)*.5+.5)*1.5);
        h = clamp(h + waterLevel * .2, 0.0, 1.0);
    }else
        h = map4(dir * 3.0+ map4(dir * 5.0) * (sin(iTime)*.5+.5)*1.5);
    
	return h;
}

vec2 map(vec3 pos){
    float l = length(pos);
    float h = height(pos / l);
    float rh = max(.45, h);
	return vec2(l - (1.0 + .2 * rh), h);
}

vec3 calcNormal( in vec3 pos )
{
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
    return normalize( e.xyy*map( pos + e.xyy ).x + 
					  e.yyx*map( pos + e.yyx ).x + 
					  e.yxy*map( pos + e.yxy ).x + 
					  e.xxx*map( pos + e.xxx ).x );
}

vec3 getGroundColor(vec3 pos, float h){
    vec3 normal = calcNormal(pos);
    float light = clamp(dot(-lightDirection, normal), 0.0, 1.0);
    
    float n = dot(normal, normalize(pos));
    
    light += pow(normal.y * .5 + .5, 2.0) * .01; // ambient
    
    vec3 col = vec3(0.0);
    
    float l = h - waterLevel;
    if (l < 0.0)
        // water
    	col = mix(vec3(0.3, 0.6, 1.0), vec3(0.0, 0.0, 1.0), clamp(pow(-l*25.0, 3.0), 0.0, 1.0));
    else{
        // land
        
        // poles
        if (abs(pos.y) > innerRadius * .9)
            col = vec3(1.0);
        else{
            if (l < 0.02)
                col = vec3(0.9, 0.85, 0.8); // sand
            else if (l < .2){
                if (n < .95)
                    col = vec3(.2, .2, .2); // rock
                else
                    col = vec3(0.0, 0.55, 0.02); // grass
            } else{
                if (n < .95)
                    col = vec3(.2); // rock
                else
                    col = vec3(1.0); // snow
            }
        }
    }
    
    return col * light;
}

vec2 raySphere(in vec3 ro, in vec3 rd, in float rad) {
	float b = 2.0 * dot(ro, rd);
    float c = dot(ro,ro) - rad*rad;
    float det = b*b - 4.0 * c;
    if (det > 0.0){
        det = sqrt(det);
    	return vec2(0.5 * (-b - det), 0.5 * (-b + det));
    }
    return vec2(-1.0, -1.0);
}

vec3 sampleAtmosphere(vec3 ro, vec3 rd, float start, float end){
	float scale = 1.0 / (atmoRadius - innerRadius);
    
    vec3 sumR = vec3(0.0);
    vec3 sumM = vec3(0.0);
    float odr = 0.0;
    float odm = 0.0;
    
	float mu = dot(rd, -lightDirection);
	float phaseM = 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + mu * mu)) / ((2.0 + g*g) * pow(1.0f + g*g - 2.0 * g * mu, 1.5));
	float phaseR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    
    float t = start;
    float slength = (end - start) * (1.0 / 6.0);
    float scaledLength = (end - start) * scale;
    vec3 pos;
    for (int i = 0; i < 6; i++){
        pos = ro + rd * t;
        
        float h = (length(pos) - innerRadius) * scale;
        
        float r = exp(-h / Hr) * scaledLength;
        float m = exp(-h / Hm) * scaledLength;
        odr += r;
        odm += m;
        
        float lodr = 0.0;
        float lodm = 0.0;
        float lt = 0.0;
        vec2 li = raySphere(pos, -lightDirection, atmoRadius);
        float llength = max(li.x, li.y) * .25;
        float scaledllength = llength * scale;
        bool f = true;
        
        for (int i = 0; i < 4; i++){
            vec3 lpos = pos + -lightDirection * lt;
            if (map(lpos).x < 0.0) { f = false; break; }
            
            float lh = (length(lpos) - innerRadius) * scale;
            float lr = exp(-lh / Hr) * scaledllength;
            float lm = exp(-lh / Hm) * scaledllength;
            lodr += lr;
            lodm += lm;
            lt += llength;
        }
        if (f) {
        	vec3 tau = betaR * (odr + lodr) + betaM * 1.1 * (odm + lodm);
            vec3 atten = exp(-tau);
            sumR += atten * r;
            sumM += atten * m;
        }
        
        t += slength;
    }
    
    return 20.0 * (sumR * betaR * phaseR + sumM * betaM * phaseM);
}

vec3 march(vec3 ro, vec3 rd){
    vec2 ray = raySphere(ro, rd, atmoRadius);
    if (ray.x < 0.0 && ray.y < 0.0) return vec3(0.0);
    
    float start = min(ray.x, ray.y);
    float end = max(ray.x, ray.y);
    
    vec3 color = vec3(0.0);
    
    float t = start;
    vec2 d;
    vec3 pos;
  	vec2 ld = vec2(1000.0, 1.0);
    for (int i = 0; i < 100; i++){
        pos = ro + rd * t;
        d = map(pos);
        
        if (d.x > ld.x && d.x > atmoRadius){
            // getting farther away, and outside the planet, exit early
            break;
        }
        
        if (d.x < 0.0){
            // hit the ground, sample ground color, blend atmo color
            color = getGroundColor(pos, d.y);
            end = t;
            break;
        }
        
        ld = d;
        t += max(d.x * .5, .01);
    }
    
    vec3 atmo = vec3(0.0);
    #if ATMOSPHERE
    atmo = sampleAtmosphere(ro, rd, start, end);
    #endif
    return color * (vec3(1.0) - atmo) + atmo;
}

vec3 tonemap(vec3 color)
{
	float white = 2.;
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = (1. + luma / (white*white)) / (1. + luma);
	return pow(color * toneMappedLuma, vec3(invgamma));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    
    vec3 dir = vec3((uv.x - .5), (uv.y - .5) * iResolution.y / iResolution.x, 1.0);
    
    dir = normalize(dir);
    
    float t = -PI * 2.0 * iMouse.x / iResolution.x + PI;
    
    vec3 campos = vec3(sin(t), 0.0, cos(t)) * (4.0 * (1.0 - (iMouse.y / iResolution.y)) + atmoRadius*1.1);
    vec3 right = vec3(sin(t + PI * .5), 0.0, cos(t + PI * .5));
    vec3 fwd = normalize(-campos);
    
    vec3 color = march(campos, normalize(right * dir.x + fwd * dir.z + vec3(0.0, dir.y, 0.0)));
    
    color = tonemap(color);
    
    fragColor = vec4(color, 1.0);
}

void mainVR(out vec4 fragColor, in vec2 fragCoord, in vec3 fragRayOri, in vec3 fragRayDir) {
    vec3 color = march(fragRayOri, fragRayDir);
    
    color = tonemap(color);
    
    fragColor = vec4(color, 1.0);
}